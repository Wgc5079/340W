#!/usr/bin/env python3
"""
multi_model_compare.py

Runs the same FinQA questions through multiple model providers and saves:
1) one predictions JSON per provider (compatible with llm_eval.py)
2) one combined CSV with per-question answers across providers

Providers supported in this script:
- openai        -> ChatGPT / OpenAI API
- gemini        -> Google Gemini API
- claude        -> Anthropic Claude API
- github_models -> GitHub Models API (good programmable replacement for Copilot-style comparison)

Recommended first run:
python multi_model_compare.py --dataset dev.json --providers openai gemini claude github_models --limit 5 --out_dir outputs

Then evaluate each provider with llm_eval.py, for example:
python llm_eval.py --dataset dev.json --predictions outputs/openai_preds.json
python llm_eval.py --dataset dev.json --predictions outputs/gemini_preds.json
python llm_eval.py --dataset dev.json --predictions outputs/claude_preds.json
python llm_eval.py --dataset dev.json --predictions outputs/github_models_preds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_PROVIDERS = {"openai", "gemini", "claude", "github_models"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def select_examples(dataset: List[Dict[str, Any]], start: int, limit: Optional[int], item_id: Optional[str]) -> List[Dict[str, Any]]:
    if item_id:
        return [x for x in dataset if x.get("id") == item_id]
    selected = dataset[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def extract_openai_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    # fallback
    try:
        out = []
        for item in response.output:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    out.append(text)
        return "\n".join(out).strip()
    except Exception:
        return str(response)


def call_openai(prompt: str, model_name: Optional[str]) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from e

    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return extract_openai_text(response)


def call_gemini(prompt: str, model_name: Optional[str]) -> str:
    try:
        from google import genai
    except Exception as e:
        raise RuntimeError("google-genai package is not installed. Run: pip install google-genai") from e

    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()


def call_claude(prompt: str, model_name: Optional[str]) -> str:
    try:
        import anthropic
    except Exception as e:
        raise RuntimeError("anthropic package is not installed. Run: pip install anthropic") from e

    model = model_name or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def call_github_models(prompt: str, model_name: Optional[str]) -> str:
    try:
        import requests
    except Exception as e:
        raise RuntimeError("requests package is not installed. Run: pip install requests") from e

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN or GITHUB_PAT before using github_models provider.")

    model = model_name or os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1")

    url = "https://models.github.ai/inference/chat/completions"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"github_models error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected GitHub Models response: {data}") from e


def ask_provider(provider: str, prompt: str, model_name: Optional[str]) -> str:
    if provider == "openai":
        return call_openai(prompt, model_name)
    if provider == "gemini":
        return call_gemini(prompt, model_name)
    if provider == "claude":
        return call_claude(prompt, model_name)
    if provider == "github_models":
        return call_github_models(prompt, model_name)
    raise ValueError(f"Unsupported provider: {provider}")


def write_combined_csv(path: Path, rows: List[Dict[str, Any]], providers: List[str]) -> None:
    fieldnames = ["id", "question", "gold_answer"] + [f"{p}_answer" for p in providers] + [f"{p}_time" for p in providers]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinQA questions across multiple model providers.")
    parser.add_argument("--dataset", required=True, help="Path to dev.json or test.json")
    parser.add_argument("--providers", nargs="+", required=True, help="Providers: openai gemini claude github_models")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save prediction files")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=None, help="How many examples to run")
    parser.add_argument("--id", dest="item_id", default=None, help="Run only one exact example id")
    parser.add_argument("--no_model_input", action="store_true", help="Use full table/text fallback instead of model_input")
    parser.add_argument("--openai_model", default=None, help="Override OpenAI model name")
    parser.add_argument("--gemini_model", default=None, help="Override Gemini model name")
    parser.add_argument("--claude_model", default=None, help="Override Claude model name")
    parser.add_argument("--github_models_model", default=None, help="Override GitHub Models model id")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers]
    bad = [p for p in providers if p not in SUPPORTED_PROVIDERS]
    if bad:
        raise ValueError(f"Unsupported providers: {bad}. Supported: {sorted(SUPPORTED_PROVIDERS)}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json(dataset_path)
    examples = select_examples(dataset, start=args.start, limit=args.limit, item_id=args.item_id)
    if not examples:
        print("No matching examples found.")
        return

    provider_to_model = {
        "openai": args.openai_model,
        "gemini": args.gemini_model,
        "claude": args.claude_model,
        "github_models": args.github_models_model,
    }

    provider_predictions: Dict[str, List[Dict[str, Any]]] = {p: [] for p in providers}
    combined_rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples, start=1):
        ex_id = ex.get("id")
        qa = ex.get("qa", {})
        question = qa.get("question", "")
        gold = qa.get("exe_ans")
        prompt = build_prompt(ex, use_model_input=not args.no_model_input)

        print(f"\n[{idx}/{len(examples)}] {ex_id}")
        combined = {
            "id": ex_id,
            "question": question,
            "gold_answer": gold,
        }

        for provider in providers:
            print(f"  -> {provider}")
            t0 = time.perf_counter()
            try:
                answer = ask_provider(provider, prompt, provider_to_model.get(provider))
                elapsed = round(time.perf_counter() - t0, 4)
                error_text = None
            except Exception as e:
                answer = f"ERROR: {e}"
                elapsed = round(time.perf_counter() - t0, 4)
                error_text = str(e)

            provider_predictions[provider].append({
                "id": ex_id,
                "model": provider,
                "predicted_answer": answer,
                "response_time": elapsed,
            })

            combined[f"{provider}_answer"] = answer
            combined[f"{provider}_time"] = elapsed

            if error_text:
                print(f"     error: {error_text}")
            else:
                short = answer.replace("\n", " ")
                if len(short) > 100:
                    short = short[:97] + "..."
                print(f"     answer: {short}")

        combined_rows.append(combined)

        for provider in providers:
            save_json(out_dir / f"{provider}_preds.json", provider_predictions[provider])

    write_combined_csv(out_dir / "combined_answers.csv", combined_rows, providers)

    print("\nDone.")
    print(f"Saved files in: {out_dir.resolve()}")
    for provider in providers:
        print(f"  - {provider}_preds.json")
    print("  - combined_answers.csv")


if __name__ == "__main__":
    main()
