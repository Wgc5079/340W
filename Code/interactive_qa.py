#!/usr/bin/env python3
"""
interactive_qa.py

Interactive FinQA helper for:
- searching examples by id or keyword
- previewing the prompt/context
- manually pasting a model answer
- saving answers in llm_eval.py-compatible JSON format

Recommended use:
1) Search in dev.json
2) Pick one example
3) Paste the answer from ChatGPT / Gemini / Copilot
4) Save to a predictions file
5) Evaluate with llm_eval.py

Example:
python interactive_qa.py --dataset dev.json --predictions chatgpt_preds.json --model_name chatgpt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def format_model_input(model_input: List[List[str]]) -> str:
    lines = []
    for item in model_input:
        if isinstance(item, list) and len(item) == 2:
            source, text = item
            lines.append(f"[{source}] {text}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def format_table(table: List[List[Any]]) -> str:
    return "\n".join(" | ".join(str(x) for x in row) for row in table)


def build_prompt(example: Dict[str, Any], use_model_input: bool = True) -> str:
    qa = example.get("qa", {})
    question = qa.get("question", "").strip()

    if use_model_input and qa.get("model_input"):
        context = format_model_input(qa["model_input"])
        context_label = "Relevant context"
    elif use_model_input and example.get("model_input"):
        context = format_model_input(example["model_input"])
        context_label = "Relevant context"
    else:
        pre_text = " ".join(example.get("pre_text", []))
        post_text = " ".join(example.get("post_text", []))
        table_text = format_table(example.get("table", []))
        context = f"Table:\n{table_text}\n\nPre-text:\n{pre_text}\n\nPost-text:\n{post_text}"
        context_label = "Document context"

    return f"""You are solving a financial reasoning question.

Use the provided context to answer the question.
Return only the final answer.
Do not explain your steps unless asked.

{context_label}:
{context}

Question:
{question}
"""


def search_examples(dataset: List[Dict[str, Any]], query: str, limit: int = 10) -> List[Dict[str, Any]]:
    q = normalize(query)
    scored = []

    for ex in dataset:
        ex_id = ex.get("id", "")
        qa = ex.get("qa", {})
        question = qa.get("question", "")
        haystack_parts = [ex_id, question]

        if qa.get("model_input"):
            for item in qa["model_input"]:
                if isinstance(item, list) and len(item) == 2:
                    haystack_parts.append(item[1])

        haystack = normalize(" ".join(haystack_parts))
        score = 0
        for token in q.split():
            if token in haystack:
                score += 1

        if q in normalize(ex_id):
            score += 100
        if q in normalize(question):
            score += 20

        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:limit]]


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if path.exists():
        data = load_json(path)
        if isinstance(data, list):
            return data
    return []


def upsert_prediction(preds: List[Dict[str, Any]], new_pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    replaced = False
    for i, pred in enumerate(preds):
        if pred.get("id") == new_pred.get("id") and pred.get("model") == new_pred.get("model"):
            preds[i] = new_pred
            replaced = True
            break
    if not replaced:
        preds.append(new_pred)
    return preds


def print_example_summary(ex: Dict[str, Any], idx: int) -> None:
    qa = ex.get("qa", {})
    print(f"\n[{idx}] {ex.get('id')}")
    print(f"Question: {qa.get('question', '')}")
    if qa.get("exe_ans") is not None:
        print(f"Gold answer (debug only): {qa.get('exe_ans')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive FinQA QA helper")
    parser.add_argument("--dataset", required=True, help="Path to FinQA dataset JSON")
    parser.add_argument("--predictions", required=True, help="Path to output predictions JSON")
    parser.add_argument("--model_name", default="chatgpt", help="Model name saved in predictions")
    parser.add_argument("--no_model_input", action="store_true", help="Use full table/text instead of model_input")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_json(dataset_path)
    preds = load_predictions(predictions_path)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Loaded predictions: {predictions_path} ({len(preds)} existing records)")
    print(f"Model name: {args.model_name}")

    while True:
        print("\nOptions:")
        print("1. Search by keyword or id")
        print("2. Open exact id")
        print("3. Show saved predictions count")
        print("4. Quit")

        choice = input("\nChoose an option: ").strip()

        if choice == "4":
            print("Goodbye.")
            break

        if choice == "3":
            print(f"Saved predictions: {len(preds)}")
            continue

        if choice == "2":
            item_id = input("Enter exact example id: ").strip()
            matches = [x for x in dataset if x.get("id") == item_id]
        elif choice == "1":
            query = input("Enter keyword or id fragment: ").strip()
            matches = search_examples(dataset, query, limit=10)
        else:
            print("Invalid option.")
            continue

        if not matches:
            print("No matches found.")
            continue

        print("\nMatches:")
        for i, ex in enumerate(matches, start=1):
            print_example_summary(ex, i)

        pick = input("\nPick a number to open (or press Enter to cancel): ").strip()
        if not pick:
            continue

        try:
            selected = matches[int(pick) - 1]
        except Exception:
            print("Invalid selection.")
            continue

        qa = selected.get("qa", {})
        prompt = build_prompt(selected, use_model_input=not args.no_model_input)

        print("\n" + "=" * 90)
        print("PROMPT")
        print("=" * 90)
        print(prompt)
        print("=" * 90)

        answer = input("\nPaste model answer here (or press Enter to cancel): ").strip()
        if not answer:
            print("Cancelled.")
            continue

        elapsed_input = input("Enter response time in seconds (or press Enter to auto-use prompt display time only): ").strip()
        if elapsed_input:
            try:
                response_time = round(float(elapsed_input), 4)
            except Exception:
                response_time = 0.0
        else:
            response_time = 0.0

        record = {
            "id": selected.get("id"),
            "model": args.model_name,
            "predicted_answer": answer,
            "response_time": response_time,
        }

        preds = upsert_prediction(preds, record)
        save_json(predictions_path, preds)

        print(f"\nSaved prediction for {selected.get('id')} to {predictions_path}")
        print("You can now run llm_eval.py on this predictions file.")


if __name__ == "__main__":
    main()
