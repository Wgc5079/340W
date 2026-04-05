#!/usr/bin/env python3
"""
ask_live_compare.py

Ask one question and send it to multiple AI providers at the same time.
Then save a CSV comparing their answers and response times.

Supported providers:
- openai         -> ChatGPT / OpenAI API
- gemini         -> Google Gemini API
- claude         -> Anthropic Claude API
- github_models  -> GitHub Models (good programmable GitHub-side comparison)

Examples:
python ask_live_compare.py --providers openai gemini claude github_models
python ask_live_compare.py --providers openai gemini --question "What is 25% of 400?"
python ask_live_compare.py --providers openai gemini claude --question "What is the net change in revenue?" --context_file prompt_context.txt --out_dir live_outputs

Notes:
- No training is needed. This script calls model APIs directly.
- If you want FinQA context, pass a text file with the table/context using --context_file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_PROVIDERS = {"openai", "gemini", "claude", "github_models"}


def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = text.strip("._-")
    if not text:
        text = "question"
    return text[:max_len]


def build_prompt(question: str, context: Optional[str]) -> str:
    if context:
        return f"""You are answering a question.

Use the provided context to answer the question.
Return a concise answer.

Context:
{context}

Question:
{question}
"""
    return f"""You are answering a question.

Return a concise answer.

Question:
{question}
"""


def extract_openai_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    try:
        chunks = []
        for item in response.output:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    except Exception:
        return str(response)


def call_openai(prompt: str, model_name: Optional[str]) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from e

    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)
    return extract_openai_text(response)


def call_gemini(prompt: str, model_name: Optional[str]) -> str:
    try:
        from google import genai
    except Exception as e:
        raise RuntimeError("google-genai package is not installed. Run: pip install google-genai") from e

    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client()
    response = client.models.generate_content(model=model, contents=prompt)
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
        max_tokens=512,
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
        raise RuntimeError("Set GITHUB_TOKEN or GITHUB_PAT before using github_models.")

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
        "messages": [{"role": "user", "content": prompt}],
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["timestamp", "question", "provider", "model_name", "response_time", "answer"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one question to multiple AI providers and compare their answers.")
    parser.add_argument("--providers", nargs="+", required=True, help="Providers: openai gemini claude github_models")
    parser.add_argument("--question", default=None, help="Question to ask. If omitted, you will be prompted.")
    parser.add_argument("--context_file", default=None, help="Optional text file with context/table/document content")
    parser.add_argument("--out_dir", default="live_outputs", help="Directory to save CSV/JSON outputs")
    parser.add_argument("--openai_model", default=None, help="Override OpenAI model name")
    parser.add_argument("--gemini_model", default=None, help="Override Gemini model name")
    parser.add_argument("--claude_model", default=None, help="Override Claude model name")
    parser.add_argument("--github_models_model", default=None, help="Override GitHub Models model name")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers]
    bad = [p for p in providers if p not in SUPPORTED_PROVIDERS]
    if bad:
        raise ValueError(f"Unsupported providers: {bad}. Supported providers: {sorted(SUPPORTED_PROVIDERS)}")

    question = args.question
    if not question:
        question = input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    context = None
    if args.context_file:
        context_path = Path(args.context_file)
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")
        context = context_path.read_text(encoding="utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_to_model = {
        "openai": args.openai_model,
        "gemini": args.gemini_model,
        "claude": args.claude_model,
        "github_models": args.github_models_model,
    }

    prompt = build_prompt(question, context)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_filename(question)

    rows: List[Dict[str, Any]] = []
    combined: Dict[str, Any] = {
        "timestamp": timestamp,
        "question": question,
        "context_file": args.context_file,
        "providers": [],
    }

    print("\nRunning providers...")
    for provider in providers:
        model_name = provider_to_model.get(provider)
        print(f"-> {provider}")
        t0 = time.perf_counter()
        try:
            answer = ask_provider(provider, prompt, model_name)
            error = None
        except Exception as e:
            answer = f"ERROR: {e}"
            error = str(e)
        elapsed = round(time.perf_counter() - t0, 4)

        short = answer.replace("\n", " ")
        if len(short) > 120:
            short = short[:117] + "..."
        print(f"   time={elapsed}s")
        print(f"   answer={short}")

        row = {
            "timestamp": timestamp,
            "question": question,
            "provider": provider,
            "model_name": model_name or "",
            "response_time": elapsed,
            "answer": answer,
        }
        rows.append(row)
        combined["providers"].append({
            "provider": provider,
            "model_name": model_name or "",
            "response_time": elapsed,
            "answer": answer,
            "error": error,
        })

    csv_path = out_dir / f"{timestamp}_{slug}_compare.csv"
    json_path = out_dir / f"{timestamp}_{slug}_compare.json"
    write_csv(csv_path, rows)
    write_json(json_path, combined)

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
