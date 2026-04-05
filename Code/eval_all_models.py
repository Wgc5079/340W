#!/usr/bin/env python3
"""
eval_all_models.py

Evaluate multiple prediction files against one FinQA dataset and produce:
1) a single summary CSV
2) a single summary JSON
3) optional detailed CSV with one row per prediction

This script reuses the same logic as llm_eval.py but evaluates all provider files
in one run.

Example:
python eval_all_models.py --dataset dev.json --predictions_dir outputs --out_dir eval_outputs

Expected prediction files in predictions_dir:
- openai_preds.json
- gemini_preds.json
- claude_preds.json
- github_models_preds.json
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

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
YES_NO_VALUES = {"yes", "no", "true", "false"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    text = normalize_text(value)
    if not text:
        return None

    text_clean = text.replace("$", "").replace("usd", "").strip()

    if text_clean.endswith("%"):
        num = safe_float(text_clean[:-1].replace(",", "").strip())
        if num is not None:
            return num / 100.0

    percent_match = re.search(r"([-+]?\d[\d,]*\.?\d*)\s*%", text_clean)
    if percent_match:
        num = safe_float(percent_match.group(1).replace(",", ""))
        if num is not None:
            return num / 100.0

    match = NUMBER_RE.search(text_clean)
    if match:
        num = safe_float(match.group(0).replace(",", ""))
        if num is not None and not (math.isnan(num) or math.isinf(num)):
            return num

    return None


def parse_possible_label(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None

    if text in YES_NO_VALUES:
        return text

    for token in YES_NO_VALUES:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return token

    return text


def is_numeric_gold(gold: Any) -> bool:
    return isinstance(gold, (int, float)) and not isinstance(gold, bool)


def answers_match(gold: Any, pred_raw: Any, abs_tol: float, rel_tol: float) -> Tuple[bool, Optional[float], Optional[str], str]:
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


def load_dataset(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
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


def evaluate_prediction_rows(dataset_by_id: Dict[str, Dict[str, Any]], predictions: List[Dict[str, Any]], abs_tol: float, rel_tol: float) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
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
        parse_failures = sum(1 for r in rows if r["reason"] in {"prediction_not_numeric", "prediction_not_text"})

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


def read_prediction_files(predictions_dir: Path) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for path in sorted(predictions_dir.glob("*_preds.json")):
        try:
            data = load_json(path)
            if isinstance(data, list):
                all_rows.extend(data)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
    return all_rows


def write_summary_csv(path: Path, summary: Dict[str, Dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "total",
        "correct",
        "accuracy",
        "avg_response_time",
        "median_response_time",
        "missing_ids",
        "parse_failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(summary.keys()):
            writer.writerow(summary[model])


def write_details_csv(path: Path, details: List[Dict[str, Any]]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def print_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    if not summary:
        print("No predictions found.")
        return

    header = (
        f"{'Model':20} {'Total':>8} {'Correct':>8} {'Accuracy':>10} "
        f"{'Avg Time':>12} {'Median Time':>12} {'Missing IDs':>12} {'Parse Fail':>12}"
    )
    print(header)
    print("-" * len(header))
    for model in sorted(summary.keys()):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all *_preds.json files in one folder.")
    parser.add_argument("--dataset", required=True, help="Path to dev.json or test.json")
    parser.add_argument("--predictions_dir", required=True, help="Folder containing *_preds.json files")
    parser.add_argument("--out_dir", default="eval_outputs", help="Where to save combined evaluation outputs")
    parser.add_argument("--abs_tol", type=float, default=1e-2, help="Absolute tolerance for numeric comparison")
    parser.add_argument("--rel_tol", type=float, default=1e-4, help="Relative tolerance for numeric comparison")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.out_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_by_id = load_dataset(dataset_path)
    all_predictions = read_prediction_files(predictions_dir)

    details, summary = evaluate_prediction_rows(
        dataset_by_id=dataset_by_id,
        predictions=all_predictions,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )

    print_summary(summary)

    save_json(out_dir / "all_models_summary.json", summary)
    write_summary_csv(out_dir / "all_models_summary.csv", summary)
    write_details_csv(out_dir / "all_models_details.csv", details)

    print(f"\nSaved:")
    print(f"  {out_dir / 'all_models_summary.json'}")
    print(f"  {out_dir / 'all_models_summary.csv'}")
    print(f"  {out_dir / 'all_models_details.csv'}")


if __name__ == "__main__":
    main()
