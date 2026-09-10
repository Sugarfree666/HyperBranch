"""Compare fixed-index QA results from two saved HyperBranch runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from eval.eval import cal_em, cal_f1


def load_result(root: Path, index: int) -> dict:
    return json.loads((root / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))


def dag_signature(result: dict) -> list[tuple]:
    return [
        (node["id"], node["question"], tuple(node.get("depends_on", [])))
        for node in result["atomic_question_dag"]["nodes"]
    ]


def leaf(result: dict) -> dict:
    return result["nodes"][-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--show-deltas", type=int, default=0)
    args = parser.parse_args()

    questions = json.loads((ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    root = ROOT / "runs" / "depo_hyperbranch" / args.dataset
    left_root, right_root = root / args.left, root / args.right
    rows: list[dict] = []
    for index in range(args.start, args.end):
        left, right = load_result(left_root, index), load_result(right_root, index)
        left_leaf, right_leaf = leaf(left), leaf(right)
        gold = str(questions[index - 1]["answer"])
        left_answer = str(left_leaf.get("answer", ""))
        right_answer = str(right_leaf.get("answer", ""))
        rows.append(
            {
                "index": index,
                "question": questions[index - 1]["question"],
                "left_answer": left_answer,
                "right_answer": right_answer,
                "left_leaf_question": left_leaf.get("rewritten_question", ""),
                "right_leaf_question": right_leaf.get("rewritten_question", ""),
                "left_f1": cal_f1([[gold]], [left_answer]),
                "right_f1": cal_f1([[gold]], [right_answer]),
                "same_dag": dag_signature(left) == dag_signature(right),
                "same_leaf_question": left_leaf.get("rewritten_question") == right_leaf.get("rewritten_question"),
                "same_first_evidence": (
                    left_leaf.get("evidence_blocks", [{}])[0].get("text")
                    == right_leaf.get("evidence_blocks", [{}])[0].get("text")
                ),
                "same_answer": left_answer == right_answer,
                "left_nodes": len(left["atomic_question_dag"]["nodes"]),
                "right_nodes": len(right["atomic_question_dag"]["nodes"]),
            }
        )

    def mean(items: list[dict], key: str) -> float:
        return sum(float(item[key]) for item in items) / len(items) if items else 0.0

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row["left_nodes"])].append(row)
    result = {
        "dataset": args.dataset,
        "range": [args.start, args.end],
        "f1": {"left": mean(rows, "left_f1"), "right": mean(rows, "right_f1")},
        "counts": {
            "same_dag": sum(row["same_dag"] for row in rows),
            "same_leaf_question": sum(row["same_leaf_question"] for row in rows),
            "same_first_evidence": sum(row["same_first_evidence"] for row in rows),
            "same_answer": sum(row["same_answer"] for row in rows),
            "right_better": sum(row["right_f1"] > row["left_f1"] for row in rows),
            "left_better": sum(row["right_f1"] < row["left_f1"] for row in rows),
        },
        "mean_dag_nodes": {"left": mean(rows, "left_nodes"), "right": mean(rows, "right_nodes")},
        "f1_by_dag_agreement": {
            "same": {
                "count": sum(row["same_dag"] for row in rows),
                "left": mean([row for row in rows if row["same_dag"]], "left_f1"),
                "right": mean([row for row in rows if row["same_dag"]], "right_f1"),
            },
            "different": {
                "count": sum(not row["same_dag"] for row in rows),
                "left": mean([row for row in rows if not row["same_dag"]], "left_f1"),
                "right": mean([row for row in rows if not row["same_dag"]], "right_f1"),
            },
        },
        "f1_by_leaf_question_agreement": {
            "same": {
                "count": sum(row["same_leaf_question"] for row in rows),
                "left": mean([row for row in rows if row["same_leaf_question"]], "left_f1"),
                "right": mean([row for row in rows if row["same_leaf_question"]], "right_f1"),
            },
            "different": {
                "count": sum(not row["same_leaf_question"] for row in rows),
                "left": mean([row for row in rows if not row["same_leaf_question"]], "left_f1"),
                "right": mean([row for row in rows if not row["same_leaf_question"]], "right_f1"),
            },
        },
        "f1_by_left_dag_size": {
            size: {"count": len(items), "left": mean(items, "left_f1"), "right": mean(items, "right_f1")}
            for size, items in sorted(buckets.items(), key=lambda item: int(item[0]))
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show_deltas:
        deltas = sorted(rows, key=lambda row: row["right_f1"] - row["left_f1"])
        selected = deltas[: args.show_deltas] + deltas[-args.show_deltas :]
        for row in selected:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
