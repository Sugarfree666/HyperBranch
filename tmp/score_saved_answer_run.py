"""Score a saved answer-only re-run against the fixed 1-based question range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from eval.eval import cal_em, cal_f1


def answer_at(root: Path, index: int) -> str:
    path = root / f"{index:05d}" / "result.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "answer" in payload:
        return str(payload["answer"] or "").strip()
    nodes = payload.get("nodes", [])
    return str(nodes[-1].get("answer", "") or "").strip() if nodes else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", default="full_1000_1")
    parser.add_argument("--start", type=int, default=900)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--show-changes", action="store_true")
    parser.add_argument("--output-dir", help="Optional directory for test_result.json and test_score.json.")
    args = parser.parse_args()

    questions = json.loads(
        (ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8")
    )
    runs_root = ROOT / "runs" / "depo_hyperbranch" / args.dataset
    candidate_root = runs_root / args.run
    baseline_root = runs_root / args.baseline
    rows = []
    for index in range(args.start, args.end):
        gold = str(questions[index - 1]["answer"])
        candidate = answer_at(candidate_root, index)
        baseline = answer_at(baseline_root, index)
        baseline_payload = json.loads(
            (baseline_root / f"{index:05d}" / "result.json").read_text(encoding="utf-8")
        ) if (baseline_root / f"{index:05d}" / "result.json").exists() else {}
        nodes = baseline_payload.get("nodes", [])
        rows.append(
            {
                "index": index,
                "question": questions[index - 1]["question"],
                "atomic_question": nodes[-1].get("rewritten_question", "") if nodes else "",
                "gold": gold,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_em": cal_em([[gold]], [candidate]),
                "candidate_f1": cal_f1([[gold]], [candidate]),
                "baseline_em": cal_em([[gold]], [baseline]),
                "baseline_f1": cal_f1([[gold]], [baseline]),
            }
        )

    def mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    result = {
        "dataset": args.dataset,
        "range": [args.start, args.end],
        "candidate": {"em": mean("candidate_em"), "f1": mean("candidate_f1")},
        "baseline": {"em": mean("baseline_em"), "f1": mean("baseline_f1")},
        "changed": sum(row["candidate"] != row["baseline"] for row in rows),
        "improved": sum(row["candidate_f1"] > row["baseline_f1"] for row in rows),
        "worse": sum(row["candidate_f1"] < row["baseline_f1"] for row in rows),
    }
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "test_result.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "test_score.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "question_file": str((ROOT / "questions" / args.dataset / "questions.json").resolve()),
                        "runs_dir": str(candidate_root.resolve()),
                        "baseline_runs_dir": str(baseline_root.resolve()),
                        "indices": list(range(args.start, args.end)),
                        "metrics": ["em", "f1"],
                    },
                    "counts": {
                        "total": len(rows),
                        "completed": sum(
                            (candidate_root / f"{index:05d}" / "result.json").exists()
                            for index in range(args.start, args.end)
                        ),
                        "missing": sum(
                            not (candidate_root / f"{index:05d}" / "result.json").exists()
                            for index in range(args.start, args.end)
                        ),
                    },
                    "overall": result["candidate"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show_changes:
        for row in rows:
            if row["candidate_f1"] != row["baseline_f1"]:
                print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
