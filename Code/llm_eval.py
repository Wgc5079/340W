#!/usr/bin/env python3
"""
llm_eval.py

Evaluate LLM / model predictions on FinQA-style datasets using gold answers
from qa["exe_ans"].

Expected dataset format:
[
  {
    "id": "...",
    "qa": {
      "question": "...",
      "exe_ans": 94.0
    }
  },
  ...
]

Expected predictions format (JSON):
[
  {
    "id": "ETR/2016/page_23.pdf-2",
    "model": "chatgpt",
    "predicted_answer": 94.0,
    "response_time": 1.23
  },
  ...
]

Also supports predictions as CSV with columns:
id,model,predicted_answer,response_time

Features:
- numeric evaluation with tolerance
- yes/no/string evaluation
- percentage-friendly parsing
- per-model summary
- optional detailed output CSV

Usage examples:
python llm_eval.py --dataset test.json --predictions chatgpt_preds.json
python llm_eval.py --dataset test.json --predictions all_models.csv --details details.csv
python llm_eval.py --dataset dev.json --predictions preds.json --abs_tol 0.01 --rel_tol 0.001
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


YES_NO_VALUES = {"yes", "no", "true", "false"}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping:
    dataset_by_id[item_id] = {
        "id": ...,
        "question": ...,
        "gold": qa["exe_ans"],
        "raw": full_item
    }
    """
    data = load_json(dataset_path)
    by_id: Dict[str, Dict[str, Any]] = {}

    for item in data:
        item_id = item.get("id")
        qa = item.get("qa", {})
        if item_id is None or "exe_ans" not in qa:
            continue

        by_id[item_id] = {
            "id": item_id,
            "question": qa.get("question"),
            "gold": qa.get("exe_ans"),
            "raw": item,
        }

    return by_id


def load_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    suffix = pred_path.suffix.lower()

    if suffix == ".json":
        preds = load_json(pred_path)
        if not isinstance(preds, list):
            raise ValueError("Prediction JSON must be a list of objects.")
        return preds

    if suffix == ".csv":
        rows: List[Dict[str, Any]] = []
        with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    raise ValueError("Predictions file must be .json or .csv")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def parse_possible_number(value: Any) -> Optional[float]:
    """
    Attempts to extract a numeric answer from raw model output.

    Handles:
    - 94
    - "94"
    - "$94"
    - "94.0"
    - "14%"
    - "The answer is 94"
    - "0.14464"
    """
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    text = normalize_text(value)
    if not text:
        return None

    # Remove currency markers/spaces but keep percent for later handling
    text_clean = text.replace("$", "").replace("usd", "").strip()

    # If the whole thing is a percent, convert to decimal
    if text_clean.endswith("%"):
        num = safe_float(text_clean[:-1].replace(",", "").strip())
        if num is not None:
            return num / 100.0

    # If it's something like "answer: 14%"
    percent_match = re.search(r"([-+]?\d[\d,]*\.?\d*)\s*%", text_clean)
    if percent_match:
        num = safe_float(percent_match.group(1).replace(",", ""))
        if num is not None:
            return num / 100.0

    # Extract first number from free text
    match = NUMBER_RE.search(text_clean)
    if match:
        num = safe_float(match.group(0).replace(",", ""))
        if num is not None and not (math.isnan(num) or math.isinf(num)):
            return num

    return None


def parse_possible_label(value: Any) -> Optional[str]:
    """
    For yes/no or other exact textual answers.
    """
    text = normalize_text(value)
    if not text:
        return None

    # Direct yes/no
    if text in YES_NO_VALUES:
        return text

    # Search inside longer text
    for token in YES_NO_VALUES:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return token

    return text


def is_numeric_gold(gold: Any) -> bool:
    return isinstance(gold, (int, float)) and not isinstance(gold, bool)


def answers_match(
    gold: Any,
    pred_raw: Any,
    abs_tol: float,
    rel_tol: float,
) -> Tuple[bool, Optional[float], Optional[str], str]:
    """
    Returns:
    (is_correct, parsed_pred_number, parsed_pred_text, reason)
    """
    if is_numeric_gold(gold):
        pred_num = parse_possible_number(pred_raw)
        if pred_num is None:
            return False, None, None, "prediction_not_numeric"

        gold_num = float(gold)
        diff = abs(pred_num - gold_num)
        allowed = max(abs_tol, rel_tol * max(abs(gold_num), 1.0))
        ok = diff <= allowed
        return ok, pred_num, None, "numeric_match" if ok else "numeric_mismatch"

    gold_text = normalize_text(gold)
    pred_text = parse_possible_label(pred_raw)

    if pred_text is None:
        return False, None, None, "prediction_not_text"

    ok = normalize_text(pred_text) == gold_text
    return ok, None, pred_text, "text_match" if ok else "text_mismatch"


def to_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def evaluate_predictions(
    dataset_by_id: Dict[str, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    per_model_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for pred in predictions:
        item_id = pred.get("id")
        model = pred.get("model", "unknown_model")
        pred_answer = pred.get("predicted_answer")
        response_time = to_float_or_none(pred.get("response_time"))

        if item_id not in dataset_by_id:
            row = {
                "id": item_id,
                "model": model,
                "question": None,
                "gold_answer": None,
                "predicted_answer": pred_answer,
                "parsed_prediction": None,
                "is_correct": False,
                "reason": "id_not_found_in_dataset",
                "response_time": response_time,
            }
            details.append(row)
            per_model_rows[model].append(row)
            continue

        example = dataset_by_id[item_id]
        gold = example["gold"]
        question = example["question"]

        is_correct, pred_num, pred_text, reason = answers_match(
            gold=gold,
            pred_raw=pred_answer,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

        parsed_prediction: Any
        if pred_num is not None:
            parsed_prediction = pred_num
        elif pred_text is not None:
            parsed_prediction = pred_text
        else:
            parsed_prediction = None

        row = {
            "id": item_id,
            "model": model,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred_answer,
            "parsed_prediction": parsed_prediction,
            "is_correct": is_correct,
            "reason": reason,
            "response_time": response_time,
        }
        details.append(row)
        per_model_rows[model].append(row)

    summary: Dict[str, Dict[str, Any]] = {}

    for model, rows in per_model_rows.items():
        total = len(rows)
        correct = sum(1 for r in rows if r["is_correct"])
        acc = correct / total if total else 0.0

        times = [r["response_time"] for r in rows if r["response_time"] is not None]
        avg_time = statistics.mean(times) if times else None
        median_time = statistics.median(times) if times else None

        missing_ids = sum(1 for r in rows if r["reason"] == "id_not_found_in_dataset")
        parse_failures = sum(
            1
            for r in rows
            if r["reason"] in {"prediction_not_numeric", "prediction_not_text"}
        )

        summary[model] = {
            "model": model,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "avg_response_time": avg_time,
            "median_response_time": median_time,
            "missing_ids": missing_ids,
            "parse_failures": parse_failures,
        }

    return details, summary


def print_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    if not summary:
        print("No predictions were evaluated.")
        return

    models = sorted(summary.keys())
    header = (
        f"{'Model':20} {'Total':>8} {'Correct':>8} {'Accuracy':>10} "
        f"{'Avg Time':>12} {'Median Time':>12} {'Missing IDs':>12} {'Parse Fail':>12}"
    )
    print(header)
    print("-" * len(header))

    for model in models:
        s = summary[model]
        avg_time = f"{s['avg_response_time']:.4f}" if s["avg_response_time"] is not None else "n/a"
        med_time = f"{s['median_response_time']:.4f}" if s["median_response_time"] is not None else "n/a"
        print(
            f"{model:20} "
            f"{s['total']:8d} "
            f"{s['correct']:8d} "
            f"{s['accuracy']*100:9.2f}% "
            f"{avg_time:>12} "
            f"{med_time:>12} "
            f"{s['missing_ids']:12d} "
            f"{s['parse_failures']:12d}"
        )


def write_details_csv(details: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "id",
        "model",
        "question",
        "gold_answer",
        "predicted_answer",
        "parsed_prediction",
        "is_correct",
        "reason",
        "response_time",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def write_summary_json(summary: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions on FinQA.")
    parser.add_argument("--dataset", required=True, help="Path to FinQA dataset JSON.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON or CSV.")
    parser.add_argument(
        "--abs_tol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for numeric comparison. Default: 0.01",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=1e-4,
        help="Relative tolerance for numeric comparison. Default: 0.0001",
    )
    parser.add_argument(
        "--details",
        default="",
        help="Optional output CSV path for per-example results.",
    )
    parser.add_argument(
        "--summary_out",
        default="",
        help="Optional output JSON path for per-model summary.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    dataset_by_id = load_dataset(dataset_path)
    predictions = load_predictions(predictions_path)

    details, summary = evaluate_predictions(
        dataset_by_id=dataset_by_id,
        predictions=predictions,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )

    print_summary(summary)

    if args.details:
        write_details_csv(details, Path(args.details))
        print(f"\nDetailed results written to: {args.details}")

    if args.summary_out:
        write_summary_json(summary, Path(args.summary_out))
        print(f"Summary JSON written to: {args.summary_out}")


if __name__ == "__main__":
    main()