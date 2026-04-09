from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current HyperBranch runs into PRoH-style generated_answer.json.")
    parser.add_argument(
        "--question-file",
        default=str(REPO_ROOT / "questions" / "agriculture" / "questions.json"),
        help="Path to the question JSON list.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(REPO_ROOT / "runs"),
        help="Directory containing per-question run folders.",
    )
    parser.add_argument("--data-source", default="", help="Result dataset name under PRoH-eval/results/.")
    parser.add_argument("--part", default="", help="Optional part suffix, matching PRoH-eval/get_score.py.")
    parser.add_argument("--ts", default="", help="Timestamp folder name. Defaults to now().")
    parser.add_argument("--limit", type=int, default=50, help="Number of leading questions to export.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    parser.add_argument(
        "--prefer-evidence-subgraph",
        action="store_true",
        help="Use evidence_subgraph evidence first instead of reasoning thoughts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(question_file: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {question_file}, got {type(payload).__name__}")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if limit <= 0:
        raise ValueError("limit must be > 0")
    return payload[start_index : start_index + limit]


def normalize_label(text: str) -> str:
    cleaned = str(text or "").strip()
    for prefix in ("<hyperedge>", "<synonyms>"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.replace("<SEP>", " / ").strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    return " ".join(cleaned.split())


def short_text(text: str, limit: int = 300) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def resolve_run_question(run_dir: Path) -> tuple[str | None, str]:
    error_payload = load_json(run_dir / "artifacts" / "error.json")
    if isinstance(error_payload, dict):
        question = error_payload.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip(), "failed"

    thought_graph = load_json(run_dir / "artifacts" / "thought_graph.json")
    if isinstance(thought_graph, dict):
        question = thought_graph.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip(), "success"

    task_frame = load_json(run_dir / "artifacts" / "task_frame.json")
    if isinstance(task_frame, dict):
        question = task_frame.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip(), "partial"

    return None, "unknown"


def discover_latest_runs(runs_dir: Path) -> dict[str, dict[str, Any]]:
    latest_by_question: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        return latest_by_question

    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime):
        question, status = resolve_run_question(run_dir)
        if not question:
            continue
        latest_by_question[question] = {
            "run_dir": run_dir,
            "status": status,
            "mtime": run_dir.stat().st_mtime,
        }
    return latest_by_question


def build_output_dir(question_file: Path, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()

    data_source = args.data_source.strip() or question_file.parent.name or question_file.stem
    part = args.part.strip()
    run_root = f"{data_source}_{part}" if part else data_source
    run_ts = args.ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    return (CURRENT_DIR / "results" / run_root / run_ts).resolve()


def extract_generation(final_answer: dict[str, Any] | None) -> str:
    if not isinstance(final_answer, dict):
        return ""
    answer = str(final_answer.get("answer", "") or "").strip()
    return f"<answer>{answer}</answer>" if answer else ""


def split_evidence_content(content: str) -> tuple[str, str]:
    sections = [part.strip() for part in str(content or "").split("\n\n") if part.strip()]
    if not sections:
        return "", ""
    if len(sections) == 1:
        return "", sections[0]
    return sections[0], "\n\n".join(sections[1:])


def build_entity_descriptions(source_node_ids: list[Any], hyperedge_text: str) -> str:
    labels: list[str] = []
    for node_id in source_node_ids:
        cleaned = normalize_label(node_id)
        if not cleaned or cleaned == hyperedge_text:
            continue
        if cleaned not in labels:
            labels.append(cleaned)
    return "; ".join(labels)


def build_reasoning_path_from_thought(thought: dict[str, Any], hyperedge_text: str) -> str:
    kind = short_text(thought.get("kind", ""), 32)
    status = short_text(thought.get("status", ""), 32)
    objective = short_text(normalize_label(thought.get("objective", "")), 140)
    content = short_text(normalize_label(thought.get("content", "")), 280)
    parts: list[str] = []
    if kind or status:
        parts.append("/".join(part for part in (kind, status) if part))
    if objective:
        parts.append(f"objective={objective}")
    if content:
        parts.append(f"thought={content}")
    if hyperedge_text and hyperedge_text.lower() not in content.lower():
        parts.append(f"hyperedge={short_text(hyperedge_text, 220)}")
    return " | ".join(parts)


def build_reasoning_path_from_evidence_item(item: dict[str, Any], hyperedge_text: str) -> str:
    parts: list[str] = []
    notes = item.get("notes", [])
    if isinstance(notes, list):
        branch_notes = [str(note).strip() for note in notes if str(note).strip().startswith("branch:")]
        if branch_notes:
            parts.extend(branch_notes[:1])
    if hyperedge_text:
        parts.append(f"hyperedge={short_text(hyperedge_text, 220)}")
    return " | ".join(parts)


def collect_retrieved_from_thought_graph(thought_graph: dict[str, Any]) -> list[dict[str, Any]]:
    thoughts = thought_graph.get("thoughts", {})
    if not isinstance(thoughts, dict):
        return []

    thought_values = [value for value in thoughts.values() if isinstance(value, dict)]
    thought_groups = [
        [thought for thought in thought_values if thought.get("kind") == "reasoning" and thought.get("status") == "verified"],
        [thought for thought in thought_values if thought.get("kind") == "reasoning" and thought.get("status") != "root"],
        [thought for thought in thought_values if thought.get("kind") == "answer"],
    ]

    exported: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for group in thought_groups:
        for thought in sorted(group, key=lambda item: str(item.get("thought_id", ""))):
            grounding = thought.get("grounding", {})
            if not isinstance(grounding, dict):
                continue
            evidence = grounding.get("evidence", [])
            if not isinstance(evidence, list):
                continue
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "") or "").strip()
                chunk_id = str(item.get("chunk_id", "") or "").strip()
                dedupe_key = chunk_id or content
                if not content or dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                hyperedge_text, chunk_text = split_evidence_content(content)
                exported.append(
                    {
                        "reasoning_path": build_reasoning_path_from_thought(thought, hyperedge_text),
                        "src_text_chunks": chunk_text or hyperedge_text or content,
                        "entity_descriptions": build_entity_descriptions(item.get("source_node_ids", []), hyperedge_text),
                        "chunk_id": chunk_id,
                        "thought_id": thought.get("thought_id"),
                        "thought_status": thought.get("status"),
                        "source_node_ids": item.get("source_node_ids", []),
                        "source_edge_ids": item.get("source_edge_ids", []),
                        "raw_content": content,
                    }
                )
        if exported:
            break

    return exported


def collect_retrieved_from_evidence_subgraph(evidence_subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = evidence_subgraph.get("evidence", [])
    if not isinstance(evidence, list):
        return []

    exported: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        chunk_id = str(item.get("chunk_id", "") or "").strip()
        dedupe_key = chunk_id or content
        if not content or dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        hyperedge_text, chunk_text = split_evidence_content(content)
        exported.append(
            {
                "reasoning_path": build_reasoning_path_from_evidence_item(item, hyperedge_text),
                "src_text_chunks": chunk_text or hyperedge_text or content,
                "entity_descriptions": build_entity_descriptions(item.get("source_node_ids", []), hyperedge_text),
                "chunk_id": chunk_id,
                "thought_id": item.get("thought_id"),
                "thought_status": "evidence_subgraph",
                "source_node_ids": item.get("source_node_ids", []),
                "source_edge_ids": item.get("source_edge_ids", []),
                "raw_content": content,
            }
        )

    return exported


def collect_retrieved(
    thought_graph: dict[str, Any] | None,
    evidence_subgraph: dict[str, Any] | None,
    *,
    prefer_evidence_subgraph: bool,
) -> list[dict[str, Any]]:
    if prefer_evidence_subgraph:
        if isinstance(evidence_subgraph, dict):
            exported = collect_retrieved_from_evidence_subgraph(evidence_subgraph)
            if exported:
                return exported
        if isinstance(thought_graph, dict):
            return collect_retrieved_from_thought_graph(thought_graph)
        return []

    if isinstance(thought_graph, dict):
        exported = collect_retrieved_from_thought_graph(thought_graph)
        if exported:
            return exported
    if isinstance(evidence_subgraph, dict):
        return collect_retrieved_from_evidence_subgraph(evidence_subgraph)
    return []


def build_record(
    question_entry: dict[str, Any],
    run_index: dict[str, dict[str, Any]],
    *,
    prefer_evidence_subgraph: bool,
) -> dict[str, Any]:
    question = str(question_entry.get("question", "")).strip()
    record: dict[str, Any] = {
        "question": question,
        "golden_answers": list(question_entry.get("golden_answers", [])),
        "context": list(question_entry.get("context", [])),
        "nhops": question_entry.get("nhops"),
        "generation": "",
        "retrieved": [],
        "run_dir": None,
        "run_status": "missing",
    }

    run_meta = run_index.get(question)
    if not run_meta:
        return record

    run_dir = Path(run_meta["run_dir"])
    final_answer = load_json(run_dir / "artifacts" / "final_answer.json")
    thought_graph = load_json(run_dir / "artifacts" / "thought_graph.json")
    evidence_subgraph = load_json(run_dir / "artifacts" / "evidence_subgraph.json")
    error_payload = load_json(run_dir / "artifacts" / "error.json")

    generation = extract_generation(final_answer if isinstance(final_answer, dict) else None)
    retrieved = collect_retrieved(
        thought_graph if isinstance(thought_graph, dict) else None,
        evidence_subgraph if isinstance(evidence_subgraph, dict) else None,
        prefer_evidence_subgraph=prefer_evidence_subgraph,
    )

    run_status = str(run_meta.get("status", "unknown"))
    if isinstance(error_payload, dict) and not generation:
        run_status = "failed"
    elif generation:
        run_status = "success"

    record.update(
        {
            "generation": generation,
            "retrieved": retrieved,
            "run_dir": str(run_dir),
            "run_status": run_status,
        }
    )
    if isinstance(error_payload, dict):
        record["run_error"] = str(error_payload.get("error_message", "") or "")
    return record


def main() -> int:
    args = parse_args()
    question_file = Path(args.question_file).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    output_dir = build_output_dir(question_file, args)

    questions = load_questions(question_file, args.start_index, args.limit)
    run_index = discover_latest_runs(runs_dir)
    records = [
        build_record(
            question_entry,
            run_index,
            prefer_evidence_subgraph=bool(args.prefer_evidence_subgraph),
        )
        for question_entry in questions
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = output_dir / "generated_answer.json"
    meta_path = output_dir / "export_meta.json"

    generated_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "question_file": str(question_file),
                "runs_dir": str(runs_dir),
                "output_dir": str(output_dir),
                "start_index": args.start_index,
                "limit": args.limit,
                "prefer_evidence_subgraph": bool(args.prefer_evidence_subgraph),
                "counts": {
                    "total": len(records),
                    "success": sum(1 for record in records if record.get("run_status") == "success"),
                    "failed": sum(1 for record in records if record.get("run_status") == "failed"),
                    "missing": sum(1 for record in records if record.get("run_status") == "missing"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_generated={generated_path}")
    print(f"saved_meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
