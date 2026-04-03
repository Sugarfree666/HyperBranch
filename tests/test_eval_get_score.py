from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_eval_get_score_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "get_score.py"
    spec = importlib.util.spec_from_file_location("eval_get_score", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvalGetScoreTest(unittest.TestCase):
    def test_build_eval_record_extracts_answer_and_retrieved_knowledge(self) -> None:
        module = load_eval_get_score_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_dir = root / "runs"
            run_dir = runs_dir / "20260402_120000_sample"
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True)

            final_answer = {
                "answer": "1979",
                "reasoning_summary": "The evidence ties the expansion to 1979.",
                "remaining_gaps": [],
            }
            thought_graph = {
                "question": "When did the expansion happen?",
                "thoughts": {
                    "th-0002": {
                        "thought_id": "th-0002",
                        "kind": "reasoning",
                        "status": "verified",
                        "grounding": {
                            "evidence": [
                                {
                                    "chunk_id": "chunk-1",
                                    "content": "The program expanded in 1979 with strong grassroots support.",
                                    "source_node_ids": ["node-1"],
                                    "source_edge_ids": ["edge-1"],
                                }
                            ]
                        },
                    },
                    "th-0003": {
                        "thought_id": "th-0003",
                        "kind": "answer",
                        "status": "completed",
                        "grounding": {
                            "evidence": [
                                {
                                    "chunk_id": "chunk-1",
                                    "content": "The program expanded in 1979 with strong grassroots support.",
                                    "source_node_ids": ["node-1"],
                                    "source_edge_ids": ["edge-1"],
                                }
                            ]
                        },
                    },
                },
            }

            (artifacts_dir / "final_answer.json").write_text(json.dumps(final_answer), encoding="utf-8")
            (artifacts_dir / "thought_graph.json").write_text(json.dumps(thought_graph), encoding="utf-8")

            run_index = module.discover_latest_runs(runs_dir)
            record = module.build_eval_record(
                {
                    "question": "When did the expansion happen?",
                    "golden_answers": ["1979"],
                    "context": ["The program expanded in 1979 with strong grassroots support."],
                    "nhops": 2,
                },
                run_index,
            )

            self.assertEqual(record["run_status"], "success")
            self.assertEqual(record["answer"], "1979")
            self.assertIn("Reasoning summary", record["generation"])
            self.assertEqual(len(record["retrieved"]), 1)
            self.assertIn("1979", record["retrieved_knowledge"])

    def test_build_eval_record_marks_missing_run(self) -> None:
        module = load_eval_get_score_module()
        record = module.build_eval_record(
            {
                "question": "Missing question?",
                "golden_answers": ["A"],
                "context": ["ctx"],
                "nhops": 3,
            },
            {},
        )
        self.assertEqual(record["run_status"], "missing")
        self.assertEqual(record["answer"], "")
        self.assertEqual(record["retrieved"], [])


if __name__ == "__main__":
    unittest.main()
