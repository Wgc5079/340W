#!/usr/bin/env python3
"""
run_predictions.py

Build prompts from FinQA examples and generate predictions in the format expected
by llm_eval.py.

Supported modes:
1) manual      -> you paste the answer for each question
2) dummy       -> returns a placeholder answer (for testing the pipeline)
3) openai_stub -> fill in your own provider code inside call_model()

Recommended workflow:
- Start with --mode manual on 5-10 questions
- Check that preds.json looks right
- Run llm_eval.py on the saved predictions
- Then replace openai_stub with your real model call

Examples:
python run_predictions.py --dataset dev.json --out preds.json --mode manual --limit 5
python run_predictions.py --dataset test.json --out preds.json --mode dummy --start 0 --limit 3
python run_predictions.py --dataset test.json --out preds.json --mode manual --id "ETR/2016/page_23.pdf-2"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_model_input(model_input: List[List[str]]) -> str:
    """
    model_input is typically a list like:
    [
      ["table_1", "the 2014 net revenue ..."],
      ["table_8", "the 2015 net revenue ..."]
    ]
    """
    lines = []
    for item in model_input:
        if isinstance(item, list) and len(item) == 2:
            source, text = item
            lines.append(f"[{source}] {text}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def format_table(table: List[List[Any]]) -> str:
    rows = []
    for row in table:
        rows.append(" | ".join(str(x) for x in row))
    return "\n".join(rows)


def build_prompt(example: Dict[str, Any], use_model_input: bool = True) -> str:
    qa = example.get("qa", {})
    question = qa.get("question", "").strip()

    if use_model_input and example.get("model_input"):
        context = format_model_input(example["model_input"])
        context_label = "Relevant context"
    elif use_model_input and qa.get("model_input"):
        context = format_model_input(qa["model_input"])
        context_label = "Relevant context"
    else:
        # fallback to full table + surrounding text if model_input is missing
        pre_text = " ".join(example.get("pre_text", []))
        post_text = " ".join(example.get("post_text", []))
        table_text = format_table(example.get("table", []))
        context = f"Table:\n{table_text}\n\nPre-text:\n{pre_text}\n\nPost-text:\n{post_text}"
        context_label = "Document context"

    prompt = f"""You are solving a financial reasoning question.

Use the provided context to answer the question.
Return only the final answer.
Do not explain your steps unless asked.

{context_label}:
{context}

Question:
{question}
"""
    return prompt


def select_examples(
    dataset: List[Dict[str, Any]],
    start: int = 0,
    limit: Optional[int] = None,
    item_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if item_id:
        matches = [x for x in dataset if x.get("id") == item_id]
        return matches

    sliced = dataset[start:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def call_model(prompt: str, mode: str, model_name: str) -> str:
    """
    Replace the openai_stub branch with your real provider call later.
    """
    if mode == "dummy":
        return "0"

    if mode == "manual":
        print("\n" + "=" * 80)
        print("PROMPT SENT TO MODEL")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        return input("Paste model answer here: ").strip()

    if mode == "openai_stub":
        # Fill this branch with your real API/provider call.
        # Example shape:
        #   client = ...
        #   response = client....
        #   return response_text
        raise NotImplementedError(
            "openai_stub mode is a placeholder. Put your real model/API call "
            "inside call_model() before using this mode."
        )

    raise ValueError(f"Unknown mode: {mode}")


def run_predictions(
    dataset_path: Path,
    out_path: Path,
    mode: str,
    model_name: str,
    start: int,
    limit: Optional[int],
    item_id: Optional[str],
    use_model_input: bool,
    append: bool,
    show_gold: bool,
) -> None:
    dataset = load_json(dataset_path)
    selected = select_examples(dataset, start=start, limit=limit, item_id=item_id)

    if not selected:
        print("No matching examples found.")
        return

    predictions: List[Dict[str, Any]] = []
    if append and out_path.exists():
        existing = load_json(out_path)
        if isinstance(existing, list):
            predictions.extend(existing)

    seen_ids = {p.get("id") for p in predictions}

    for idx, ex in enumerate(selected, start=1):
        ex_id = ex.get("id")
        qa = ex.get("qa", {})
        question = qa.get("question", "")
        gold = qa.get("exe_ans")

        if ex_id in seen_ids:
            print(f"Skipping {ex_id} because it is already in {out_path.name}")
            continue

        print(f"\n[{idx}/{len(selected)}] ID: {ex_id}")
        print(f"Question: {question}")
        if show_gold:
            print(f"(Debug) Gold answer: {gold}")

        prompt = build_prompt(ex, use_model_input=use_model_input)

        start_time = time.perf_counter()
        raw_answer = call_model(prompt, mode=mode, model_name=model_name)
        end_time = time.perf_counter()

        pred = {
            "id": ex_id,
            "model": model_name,
            "predicted_answer": raw_answer,
            "response_time": round(end_time - start_time, 4),
        }
        predictions.append(pred)
        seen_ids.add(ex_id)

        save_json(out_path, predictions)
        print(f"Saved answer for {ex_id} -> {out_path}")

    print("\nDone.")
    print(f"Predictions saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinQA predictions and save them for llm_eval.py")
    parser.add_argument("--dataset", required=True, help="Path to FinQA JSON dataset")
    parser.add_argument("--out", required=True, help="Path to output predictions JSON")
    parser.add_argument("--mode", choices=["manual", "dummy", "openai_stub"], default="manual")
    parser.add_argument("--model_name", default="chatgpt", help="Name stored in predictions JSON")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to run")
    parser.add_argument("--id", dest="item_id", default=None, help="Run only one specific example id")
    parser.add_argument("--no_model_input", action="store_true", help="Do not use model_input; use full table/text fallback")
    parser.add_argument("--append", action="store_true", help="Append to existing output file if it exists")
    parser.add_argument("--show_gold", action="store_true", help="Show gold answer during debugging")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run_predictions(
        dataset_path=dataset_path,
        out_path=out_path,
        mode=args.mode,
        model_name=args.model_name,
        start=args.start,
        limit=args.limit,
        item_id=args.item_id,
        use_model_input=not args.no_model_input,
        append=args.append,
        show_gold=args.show_gold,
    )


if __name__ == "__main__":
    main()
