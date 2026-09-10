"""Re-answer saved HyperBranch leaf nodes without changing their DAG or evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT)]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hyper_branch.client import OpenAIClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", default="full_test1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, help="Exclusive index.")
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit 1-based question indices.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--include-previous-answer",
        action="store_true",
        help="Provide the saved leaf answer as a candidate for evidence-based verification.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/atomic_answer.md",
        help="Prompt path, relative to the project root unless absolute.",
    )
    args = parser.parse_args()
    if args.indices is None and (args.start is None or args.end is None):
        parser.error("provide --indices or both --start and --end")

    config = yaml.safe_load((PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8"))
    questions = json.loads((PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    client = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    source_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.source_run
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.run_id

    indices = args.indices or range(args.start, args.end)
    for index in indices:
        output_path = output_dir / f"{index:05d}" / "result.json"
        if args.resume and output_path.exists():
            continue
        try:
            source_path = source_dir / f"{index:05d}" / "result.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            leaf = source["nodes"][-1]
            question = questions[index - 1]["question"].strip()
            prompt_input = {
                "original_question": question,
                "atomic_question": leaf["rewritten_question"],
                # The saved leaf question has dependency placeholders substituted already.
                "dependency_context": [],
                "evidence_blocks": leaf["evidence_blocks"],
            }
            if args.include_previous_answer:
                prompt_input["previous_answer"] = leaf["answer"]
            response = client.chat_json(
                prompt,
                json.dumps(prompt_input, ensure_ascii=False, indent=2),
                max_tokens=900,
            )
            answer = str(response["answer"]).strip()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "source_result": str(source_path),
                        "question": question,
                        "gold_answer": questions[index - 1].get("answer"),
                        "leaf_node_id": leaf["id"],
                        "atomic_question": leaf["rewritten_question"],
                        "previous_answer": leaf["answer"],
                        "answer": answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {leaf['answer']} -> {answer}", flush=True)
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
