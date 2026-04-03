from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval import cal_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GoT-HGs runs with F1, R-S, and G-E.")
    parser.add_argument("--question-file", default="questions/agriculture/questions.json", help="Path to the question JSON list.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing per-question run folders.")
    parser.add_argument("--limit", type=int, default=50, help="Number of leading questions to evaluate.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to eval/results/<dataset>/<timestamp>.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for per-sample scoring. Keep 1 if using G-E online.")
    parser.add_argument("--skip-rsim", action="store_true", help="Skip R-S evaluation.")
    parser.add_argument("--skip-gen", action="store_true", help="Skip G-E evaluation.")
    return parser.parse_args()


def load_questions(question_file: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {question_file}, got {type(payload).__name__}")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if limit <= 0:
        raise ValueError("limit must be > 0")
    return payload[start_index : start_index + limit]


def build_output_dir(question_file: Path, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir)
    dataset_name = question_file.parent.name or question_file.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CURRENT_DIR / "results" / dataset_name / timestamp


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def extract_generation(final_answer: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(final_answer, dict):
        return "", ""
    answer = str(final_answer.get("answer", "") or "").strip()
    reasoning_summary = str(final_answer.get("reasoning_summary", "") or "").strip()
    remaining_gaps = final_answer.get("remaining_gaps", [])

    segments = [answer] if answer else []
    if reasoning_summary:
        segments.append(f"Reasoning summary: {reasoning_summary}")
    if isinstance(remaining_gaps, list) and remaining_gaps:
        cleaned_gaps = [str(item).strip() for item in remaining_gaps if str(item).strip()]
        if cleaned_gaps:
            segments.append("Remaining gaps: " + "; ".join(cleaned_gaps))

    return answer, "\n\n".join(segments).strip()


def _collect_evidence_from_thought(thought: dict[str, Any], sink: list[dict[str, Any]], seen_keys: set[str]) -> None:
    grounding = thought.get("grounding", {})
    if not isinstance(grounding, dict):
        return
    evidence = grounding.get("evidence", [])
    if not isinstance(evidence, list):
        return

    for item in evidence:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        chunk_id = str(item.get("chunk_id", "") or "").strip()
        if not content:
            continue
        dedupe_key = chunk_id or content
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        sink.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "source_node_ids": item.get("source_node_ids", []),
                "source_edge_ids": item.get("source_edge_ids", []),
                "thought_id": thought.get("thought_id"),
                "thought_status": thought.get("status"),
            }
        )


def extract_retrieved_knowledge(thought_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(thought_graph, dict):
        return []
    thoughts = thought_graph.get("thoughts", {})
    if not isinstance(thoughts, dict):
        return []

    seen_keys: set[str] = set()
    retrieved: list[dict[str, Any]] = []

    answer_thoughts = [thought for thought in thoughts.values() if isinstance(thought, dict) and thought.get("kind") == "answer"]
    verified_reasoning = [
        thought
        for thought in thoughts.values()
        if isinstance(thought, dict) and thought.get("kind") == "reasoning" and thought.get("status") == "verified"
    ]
    fallback_reasoning = [
        thought
        for thought in thoughts.values()
        if isinstance(thought, dict) and thought.get("kind") == "reasoning"
    ]

    for thought_group in (answer_thoughts, verified_reasoning, fallback_reasoning):
        for thought in sorted(thought_group, key=lambda item: str(item.get("thought_id", ""))):
            _collect_evidence_from_thought(thought, retrieved, seen_keys)
        if retrieved:
            break

    return retrieved


def extract_retrieved_from_evidence_subgraph(evidence_subgraph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(evidence_subgraph, dict):
        return []
    evidence_items = evidence_subgraph.get("evidence", [])
    if not isinstance(evidence_items, list):
        return []

    retrieved: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        chunk_id = str(item.get("chunk_id", "") or "").strip()
        if not content:
            continue
        dedupe_key = chunk_id or content
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "source_node_ids": item.get("source_node_ids", []),
                "source_edge_ids": item.get("source_edge_ids", []),
                "thought_id": item.get("thought_id"),
                "thought_status": "evidence_subgraph",
            }
        )
    return retrieved


def build_eval_record(question_entry: dict[str, Any], run_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    question = str(question_entry.get("question", "")).strip()
    record: dict[str, Any] = {
        "question": question,
        "golden_answers": list(question_entry.get("golden_answers", [])),
        "context": list(question_entry.get("context", [])),
        "nhops": question_entry.get("nhops"),
        "run_dir": None,
        "run_status": "missing",
        "answer": "",
        "generation": "",
        "retrieved": [],
        "retrieved_knowledge": "",
    }

    run_meta = run_index.get(question)
    if not run_meta:
        return record

    run_dir = Path(run_meta["run_dir"])
    final_answer = load_json(run_dir / "artifacts" / "final_answer.json")
    thought_graph = load_json(run_dir / "artifacts" / "thought_graph.json")
    evidence_subgraph = load_json(run_dir / "artifacts" / "evidence_subgraph.json")
    error_payload = load_json(run_dir / "artifacts" / "error.json")

    answer, generation = extract_generation(final_answer if isinstance(final_answer, dict) else None)
    retrieved = extract_retrieved_knowledge(thought_graph if isinstance(thought_graph, dict) else None)
    if not retrieved:
        retrieved = extract_retrieved_from_evidence_subgraph(evidence_subgraph if isinstance(evidence_subgraph, dict) else None)
    knowledge_text = "\n\n".join(item["content"] for item in retrieved).strip()

    run_status = str(run_meta.get("status", "unknown"))
    if isinstance(error_payload, dict) and not answer:
        run_status = "failed"
    elif answer:
        run_status = "success"

    record.update(
        {
            "run_dir": str(run_dir),
            "run_status": run_status,
            "answer": answer,
            "generation": generation or answer,
            "retrieved": retrieved,
            "retrieved_knowledge": knowledge_text,
        }
    )
    if isinstance(error_payload, dict):
        record["run_error"] = str(error_payload.get("error_message", "") or "")

    return record


def _get_rsim_fn():
    from eval_r import cal_rsim

    return cal_rsim


def _get_gen_fn():
    from eval_g import cal_gen

    return cal_gen


def evaluate_one(record: dict[str, Any], use_rsim: bool, use_gen: bool) -> dict[str, Any]:
    answer = record.get("answer", "") or ""
    gold_answers = record.get("golden_answers", [])
    f1_score = cal_f1([gold_answers], [answer]) if gold_answers else 0.0
    record["f1"] = float(f1_score)

    if use_rsim:
        rsim_fn = _get_rsim_fn()
        dedup_context = list(dict.fromkeys(str(item) for item in record.get("context", []) if str(item).strip()))
        knowledge = str(record.get("retrieved_knowledge", "") or "").strip()
        record["r_s"] = float(rsim_fn(["\n".join(dedup_context)], [knowledge])) if knowledge else 0.0
    else:
        record["r_s"] = None

    if use_gen:
        generation = str(record.get("generation", "") or "").strip()
        if not generation:
            record["g_e"] = 0.0
            record["g_e_exp"] = {"status": "empty_generation"}
        else:
            try:
                gen_fn = _get_gen_fn()
                gen_score = gen_fn(record["question"], gold_answers, generation, f1_score)
                record["g_e"] = float(gen_score["score"])
                record["g_e_exp"] = gen_score["explanation"]
            except Exception as exc:
                record["g_e"] = None
                record["g_e_exp"] = {"status": "error", "message": str(exc)}
    else:
        record["g_e"] = None

    # Backward-compatible aliases for older tooling.
    record["rsim"] = record["r_s"]
    record["gen"] = record["g_e"]
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        return mean(values) if values else None

    by_nhops: dict[str, dict[str, Any]] = {}
    for record in records:
        nhops = record.get("nhops")
        if nhops is None:
            continue
        bucket = by_nhops.setdefault(str(nhops), {"count": 0, "f1": [], "r_s": [], "g_e": []})
        bucket["count"] += 1
        for metric in ("f1", "r_s", "g_e"):
            value = record.get(metric)
            if value is not None:
                bucket[metric].append(float(value))

    summary = {
        "counts": {
            "total": len(records),
            "success": sum(1 for record in records if record.get("run_status") == "success"),
            "failed": sum(1 for record in records if record.get("run_status") == "failed"),
            "missing": sum(1 for record in records if record.get("run_status") == "missing"),
        },
        "overall": {
            "f1": avg("f1"),
            "r_s": avg("r_s"),
            "g_e": avg("g_e"),
        },
        "by_nhops": {},
    }

    for nhops, bucket in sorted(by_nhops.items(), key=lambda item: int(item[0])):
        summary["by_nhops"][nhops] = {
            "count": bucket["count"],
            "f1": mean(bucket["f1"]) if bucket["f1"] else None,
            "r_s": mean(bucket["r_s"]) if bucket["r_s"] else None,
            "g_e": mean(bucket["g_e"]) if bucket["g_e"] else None,
        }

    # Backward-compatible aliases for older tooling.
    summary["overall"]["rsim"] = summary["overall"]["r_s"]
    summary["overall"]["gen"] = summary["overall"]["g_e"]
    return summary


def save_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = output_dir / "generated_answer.json"
    result_path = output_dir / "test_result.json"
    score_path = output_dir / "test_score.json"

    generated_payload = [
        {
            "question": record["question"],
            "golden_answers": record["golden_answers"],
            "context": record["context"],
            "nhops": record.get("nhops"),
            "run_dir": record.get("run_dir"),
            "run_status": record.get("run_status"),
            "answer": record.get("answer"),
            "generation": record.get("generation"),
            "retrieved": record.get("retrieved"),
            "retrieved_knowledge": record.get("retrieved_knowledge"),
        }
        for record in records
    ]

    generated_path.write_text(json.dumps(generated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    score_path.write_text(json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    question_file = Path(args.question_file).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    output_dir = build_output_dir(question_file, args.output_dir).resolve()

    questions = load_questions(question_file, args.start_index, args.limit)
    run_index = discover_latest_runs(runs_dir)
    records = [build_eval_record(question_entry, run_index) for question_entry in questions]

    use_rsim = not args.skip_rsim
    use_gen = not args.skip_gen
    workers = max(1, int(args.workers))

    if workers == 1:
        scored_records = [evaluate_one(record, use_rsim=use_rsim, use_gen=use_gen) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scored_records = list(
                executor.map(
                    lambda record: evaluate_one(record, use_rsim=use_rsim, use_gen=use_gen),
                    records,
                )
            )

    summary = summarize(scored_records)
    meta = {
        "question_file": str(question_file),
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "limit": args.limit,
        "skip_rsim": args.skip_rsim,
        "skip_gen": args.skip_gen,
        "workers": workers,
    }
    save_outputs(output_dir, scored_records, summary, meta)

    print(f"saved_generated={output_dir / 'generated_answer.json'}")
    print(f"saved_results={output_dir / 'test_result.json'}")
    print(f"saved_scores={output_dir / 'test_score.json'}")
    print(f"F1={summary['overall']['f1'] if summary['overall']['f1'] is not None else 'N/A'}")
    print(f"R-S={summary['overall']['r_s'] if summary['overall']['r_s'] is not None else 'N/A'}")
    print(f"G-E={summary['overall']['g_e'] if summary['overall']['g_e'] is not None else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
