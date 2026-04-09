from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval import cal_em, cal_f1


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PRoH-style generated_answer.json with R-S only.")
    parser.add_argument("--data_source", default="agriculture")
    parser.add_argument("--part", default="")
    parser.add_argument("--ts", default="", help="Run timestamp folder under PRoH-eval/results/.")
    parser.add_argument("--input-file", default="", help="Explicit path to generated_answer.json.")
    parser.add_argument("--output-dir", default="", help="Explicit directory for test_result.json/test_score.json.")
    return parser.parse_args()


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    data_source = args.data_source.strip()
    part = args.part.strip()
    run_root = f"{data_source}_{part}" if part else data_source
    run_ts = args.ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    return (CURRENT_DIR / "results" / run_root / run_ts).resolve()


def build_knowledge(record: dict) -> str:
    chunks: list[str] = []
    for item in record.get("retrieved", []):
        if not isinstance(item, dict):
            continue
        for field in ("reasoning_path", "src_text_chunks", "entity_descriptions"):
            text = str(item.get(field, "") or "").strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def evaluate_one(record: dict) -> dict:
    from eval_r import cal_rsim

    generation = str(record.get("generation", "") or "").strip()
    answer = generation
    if "<answer>" in generation and "</answer>" in generation:
        try:
            answer = generation.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        except Exception:
            answer = generation

    gold_answers = list(record.get("golden_answers", []))
    context_items = []
    for item in record.get("context", []):
        text = str(item).strip()
        if text and text not in context_items:
            context_items.append(text)

    knowledge = build_knowledge(record)
    record["em"] = float(cal_em([gold_answers], [answer])) if gold_answers else 0.0
    record["f1"] = float(cal_f1([gold_answers], [answer])) if gold_answers else 0.0
    record["rsim"] = float(cal_rsim(["\n".join(context_items)], [knowledge])) if knowledge else 0.0
    record["knowledge"] = knowledge
    return record


def main() -> int:
    args = parse_args()
    if args.input_file:
        input_file = Path(args.input_file).resolve()
        run_dir = Path(args.output_dir).resolve() if args.output_dir else input_file.parent
    else:
        run_dir = build_run_dir(args)
        input_file = run_dir / "generated_answer.json"
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    records = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("generated_answer.json must be a list of records")

    scored_records = [evaluate_one(dict(record)) for record in records]
    total = len(scored_records)
    overall_em = sum(record["em"] for record in scored_records) / total if total else 0.0
    overall_f1 = sum(record["f1"] for record in scored_records) / total if total else 0.0
    overall_rsim = sum(record["rsim"] for record in scored_records) / total if total else 0.0

    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "test_result.json"
    score_path = run_dir / "test_score.json"

    result_path.write_text(json.dumps(scored_records, ensure_ascii=False, indent=2), encoding="utf-8")
    score_path.write_text(
        json.dumps(
            {
                "overall_em": overall_em,
                "overall_f1": overall_f1,
                "overall_rsim": overall_rsim,
                "count": total,
                "input_file": str(input_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_results={result_path}")
    print(f"saved_scores={score_path}")
    print(f"Overall EM: {overall_em:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Overall R-Sim: {overall_rsim:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
