"""Estimate answer-prompt headroom from saved HotpotQA leaf evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.eval import cal_em, cal_f1, normalize_answer


POLAR = re.compile(
    r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b",
    re.I,
)
COMPARISON = re.compile(
    r"\b(more|less|older|younger|first|last|higher|lower|further|farther|"
    r"larger|smaller|greater|most|least|earlier|later|longer|shorter)\b",
    re.I,
)


def question_type(question: str) -> str:
    if POLAR.match(question.strip()):
        return "polar"
    if COMPARISON.search(question):
        return "comparison"
    return "other"


def summary_by_type(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    return {
        kind: {
            "n": sum(row["type"] == kind for row in rows),
            "loss": sum(1 - float(row["f1"]) for row in rows if row["type"] == kind),
            "zero_f1": sum(row["type"] == kind and row["f1"] == 0 for row in rows),
        }
        for kind in ("polar", "comparison", "other")
    }


def main() -> None:
    questions = json.loads((ROOT / "questions/hotpotqa/questions.json").read_text(encoding="utf-8"))
    run_root = ROOT / "runs/depo_hyperbranch/hotpotqa/full_1000"
    rows: list[dict[str, object]] = []
    for index, item in enumerate(questions, start=1):
        result = json.loads((run_root / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))
        leaf = result["nodes"][-1]
        gold = str(item["answer"])
        predicted = str(leaf.get("answer", ""))
        evidence = " ".join(str(block.get("text", "")) for block in leaf.get("evidence_blocks", []))
        gold_normalized = normalize_answer(gold)
        source_titles = [
            str(block.get("title", ""))
            for block in leaf.get("evidence_blocks", [])
            if gold_normalized and gold_normalized in normalize_answer(str(block.get("text", "")))
        ]
        rows.append(
            {
                "index": index,
                "question": item["question"],
                "atomic_question": leaf.get("rewritten_question", ""),
                "gold": gold,
                "predicted": predicted,
                "em": cal_em([[gold]], [predicted]),
                "f1": cal_f1([[gold]], [predicted]),
                "type": question_type(str(item["question"])),
                "atomic_type": question_type(str(leaf.get("rewritten_question", ""))),
                "predicted_yes_no": normalize_answer(predicted) in {"yes", "no"},
                "gold_yes_no": normalize_answer(gold) in {"yes", "no"},
                "gold_in_leaf_evidence": bool(source_titles),
                "titles": source_titles,
            }
        )

    misses = [row for row in rows if float(row["f1"]) < 1]
    direct = [row for row in misses if row["gold_in_leaf_evidence"]]
    def compact_stats(selected: list[dict[str, object]]) -> dict[str, float | int]:
        return {
            "n": len(selected),
            "f1_loss": sum(1 - float(row["f1"]) for row in selected),
            "zero_f1": sum(float(row["f1"]) == 0 for row in selected),
            "gold_in_leaf_evidence": sum(bool(row["gold_in_leaf_evidence"]) for row in selected),
            "direct_evidence_loss": sum(
                1 - float(row["f1"]) for row in selected if row["gold_in_leaf_evidence"]
            ),
        }
    nonpolar_yes_no = [
        row
        for row in misses
        if row["atomic_type"] != "polar" and row["predicted_yes_no"]
    ]
    comparison_errors = [row for row in misses if row["atomic_type"] == "comparison"]
    polar_leaf_nonpolar_original = [
        row
        for row in misses
        if row["atomic_type"] == "polar" and row["type"] != "polar"
    ]
    if "--comparison-only" in sys.argv:
        print(
            json.dumps(
                [
                    {
                        key: row[key]
                        for key in (
                            "index",
                            "question",
                            "atomic_question",
                            "gold",
                            "predicted",
                            "f1",
                            "gold_in_leaf_evidence",
                            "titles",
                        )
                    }
                    for row in comparison_errors
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    total_f1 = sum(float(row["f1"]) for row in rows)
    report = {
        "overall": {
            "n": len(rows),
            "em": sum(float(row["em"]) for row in rows) / len(rows),
            "f1": total_f1 / len(rows),
            "f1_points_needed_for_71_1": 711 - total_f1,
        },
        "all_incorrect": {
            "n": len(misses),
            "f1_loss": sum(1 - float(row["f1"]) for row in misses),
            "by_type": summary_by_type(misses),
        },
        "gold_explicit_in_leaf_evidence": {
            "n": len(direct),
            "f1_recoverable_upper_bound": sum(1 - float(row["f1"]) for row in direct),
            "by_type": summary_by_type(direct),
            "zero_f1_n": sum(float(row["f1"]) == 0 for row in direct),
        },
        "answer_layer_candidate_categories": {
            "nonpolar_leaf_answered_yes_no": compact_stats(nonpolar_yes_no),
            "comparison_leaf_errors": compact_stats(comparison_errors),
            "polar_leaf_for_nonpolar_original": compact_stats(polar_leaf_nonpolar_original),
        },
        "examples_nonpolar_leaf_answered_yes_no": [
            {
                key: row[key]
                for key in ("index", "question", "atomic_question", "gold", "predicted", "titles")
            }
            for row in nonpolar_yes_no
        ][:30],
        "examples_comparison_leaf_errors": [
            {
                key: row[key]
                for key in ("index", "question", "atomic_question", "gold", "predicted", "titles")
            }
            for row in comparison_errors
            if row["gold_in_leaf_evidence"]
        ][:30],
        "examples_zero_f1_direct": [
            {
                key: row[key]
                for key in ("index", "question", "atomic_question", "gold", "predicted", "type", "titles")
            }
            for row in direct
            if float(row["f1"]) == 0
        ][:60],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
