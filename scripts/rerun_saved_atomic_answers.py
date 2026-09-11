"""Re-answer a saved DAG while reusing every node's retrieved evidence."""

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
from hyper_branch.pipeline import (
    _answer_dependency_context,
    _rewrite_question,
    _topological_order,
)


def _saved_dag(source: dict[str, object]) -> dict[str, object]:
    dag = source.get("atomic_question_dag")
    if isinstance(dag, dict):
        return dag
    dag_path = source.get("source_dag")
    if isinstance(dag_path, str):
        payload = json.loads(Path(dag_path).read_text(encoding="utf-8"))
        dag = payload.get("atomic_question_dag")
        if isinstance(dag, dict):
            return dag
    raise ValueError("Saved result does not contain an atomic_question_dag")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, help="Exclusive index.")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()
    if args.indices is None and (args.start is None or args.end is None):
        parser.error("provide --indices or both --start and --end")

    config = yaml.safe_load((PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8"))
    questions = json.loads((PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    prompt = (PROJECT_ROOT / "prompts" / "atomic_answer.md").read_text(encoding="utf-8").strip()
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
            dag = _saved_dag(source)
            saved_nodes = {node["id"]: node for node in source["nodes"]}
            answers: dict[str, dict[str, object]] = {}
            output_nodes = []
            for node in _topological_order(dag["nodes"]):
                dependency_answers = [
                    answers[node_id] for node_id in node.get("depends_on", [])
                ]
                atomic_question, _ = _rewrite_question(
                    node["question"], dependency_answers
                )
                saved = saved_nodes[node["id"]]
                dependency_context = _answer_dependency_context(dependency_answers)
                response = client.chat_json(
                    prompt,
                    json.dumps(
                        {
                            "original_question": questions[index - 1]["question"],
                            "atomic_question": atomic_question,
                            "dependency_context": dependency_context,
                            "evidence_blocks": saved["evidence_blocks"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    max_tokens=900,
                )
                answer = str(response["answer"]).strip()
                answers[node["id"]] = {
                    "node_id": node["id"],
                    "question": atomic_question,
                    "entities": saved["entities"],
                    "entity_ids": saved.get("entity_ids", {}),
                    "evidence_blocks": saved["evidence_blocks"],
                    "answer": answer,
                }
                output_nodes.append(
                    {
                        "id": node["id"],
                        "rewritten_question": atomic_question,
                        "entities": saved["entities"],
                        "entity_ids": saved.get("entity_ids", {}),
                        "evidence_blocks": saved["evidence_blocks"],
                        "answer": answer,
                    }
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "topic_entities": source["topic_entities"],
                        "atomic_question_dag": dag,
                        "topic_entity_ids": source.get("topic_entity_ids", {}),
                        "nodes": output_nodes,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {output_nodes[-1]['answer']}", flush=True)
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
