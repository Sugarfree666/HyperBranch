from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact summary from eval/test_score.json.")
    parser.add_argument("--score-file", required=True, help="Path to test_score.json produced by eval/get_score.py")
    return parser.parse_args()


def fmt_score(value: float | None) -> str:
    return f"{value * 100:.2f}" if value is not None else "N/A"


def main() -> int:
    args = parse_args()
    score_file = Path(args.score_file).resolve()
    payload = json.loads(score_file.read_text(encoding="utf-8"))

    meta = payload.get("meta", {})
    counts = payload.get("counts", {})
    overall = payload.get("overall", {})
    by_nhops = payload.get("by_nhops", {})

    print(f"score_file={score_file}")
    print(f"question_file={meta.get('question_file', 'N/A')}")
    print(f"runs_dir={meta.get('runs_dir', 'N/A')}")
    print(
        "counts="
        f"total:{counts.get('total', 0)} "
        f"success:{counts.get('success', 0)} "
        f"failed:{counts.get('failed', 0)} "
        f"missing:{counts.get('missing', 0)}"
    )
    print(
        "overall="
        f"F1:{fmt_score(overall.get('f1'))} "
        f"R-S:{fmt_score(overall.get('r_s'))} "
        f"G-E:{fmt_score(overall.get('g_e'))}"
    )

    for nhops, metrics in sorted(by_nhops.items(), key=lambda item: int(item[0])):
        print(
            f"nhops={nhops} "
            f"count={metrics.get('count', 0)} "
            f"F1:{fmt_score(metrics.get('f1'))} "
            f"R-S:{fmt_score(metrics.get('r_s'))} "
            f"G-E:{fmt_score(metrics.get('g_e'))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
