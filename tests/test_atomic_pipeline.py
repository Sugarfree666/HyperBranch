from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from hyper_branch.pipeline import HyperBranchPipeline, _evidence_blocks


class AtomicPipelineTest(unittest.TestCase):
    def test_evidence_blocks_keep_top_ranked_unique_source_chunks_only(self) -> None:
        hyperedges = [
            {
                "text": f"H{index}",
                "chunks": [(f"source-{index}", f"Evidence for H{index}")],
                "first_hop_texts": [],
            }
            for index in range(1, 11)
        ]
        hyperedges.extend(
            [
                {
                    "text": "late-existing-source",
                    "chunks": [("source-1", "Evidence for H1")],
                    "first_hop_texts": ["unused-first-hop"],
                },
                {
                    "text": "eleventh-source",
                    "chunks": [("source-11", "Evidence for H11")],
                    "first_hop_texts": [],
                },
            ]
        )

        blocks = _evidence_blocks(hyperedges)

        self.assertEqual(
            [block["chunk_id"] for block in blocks],
            [f"C{index}" for index in range(1, 11)],
        )
        self.assertEqual(blocks[0], {"chunk_id": "C1", "title": "", "text": "Evidence for H1"})
        self.assertTrue(all(set(block) == {"chunk_id", "title", "text"} for block in blocks))

    def test_pipeline_rewrites_dependencies_and_merges_all_ancestor_pools(self) -> None:
        database = RecordingDatabase(
            pools=[
                {"H-original": set()},
                {"H-q1": set()},
                {"H-q2": {"H-q1"}},
                {"H-q3": {"H-q2"}},
            ]
        )
        client = RecordingClient([["A"], ["B"], ["C"]], ["B", "C", "Done"])
        dag = {
            "nodes": [
                {"id": "q3", "question": "What follows q2's answer?", "depends_on": ["q2"]},
                {"id": "q1", "question": "Who is linked to A?", "depends_on": []},
                {"id": "q2", "question": "Where was q1's answer recorded?", "depends_on": ["q1"]},
            ]
        }

        with (
            patch("hyper_branch.pipeline.HypergraphDatabase", return_value=database),
            patch("hyper_branch.pipeline.OpenAIClient", return_value=client),
        ):
            pipeline = HyperBranchPipeline(
                Path("unused"),
                model="test-model",
                embedding_model="test-embedding-model",
                timeout_seconds=120,
                temperature=0.2,
                api_key="test-key",
            )
            result = pipeline.run("What eventually follows A?", dag, ["A"])

        self.assertEqual(
            [item["question"] for item in result["atomic_answers"]],
            ["Who is linked to A?", "Where was B recorded?", "What follows C?"],
        )
        self.assertEqual(result["final_answer"], {"answer": "Done"})
        self.assertEqual(result["topic_entity_ids"], {"A": ["A-id"]})
        self.assertEqual(result["atomic_answers"][0]["entities"], ["A"])
        self.assertEqual(result["atomic_answers"][0]["entity_ids"], {"A": ["A-id"]})
        self.assertTrue(result["atomic_answers"][0]["evidence_blocks"])
        self.assertEqual(database.original_chunk_calls, [("What eventually follows A?", 5)])
        self.assertEqual(database.anchor_calls, [["A"], ["B"], ["C"]])
        self.assertEqual(
            set(database.rank_calls[-1]["candidates"]),
            {"H-original", "H-q1", "H-q2", "H-q3"},
        )
        self.assertEqual(
            client.entity_questions,
            ["Who is linked to A?", "Where was B recorded?", "What follows C?"],
        )
        dependency = client.chat_calls[1]["dependency_context"][0]
        self.assertEqual(dependency["answer"], "B")
        self.assertTrue(dependency["entities"])
        self.assertTrue(dependency["evidence_blocks"])
        self.assertNotIn("entity_ids", dependency)
        self.assertTrue(client.chat_calls[1]["evidence_blocks"])
        self.assertTrue(
            all(
                set(block) == {"chunk_id", "title", "text"}
                for block in client.chat_calls[1]["evidence_blocks"]
            )
        )

class RecordingDatabase:
    def __init__(self, pools: list[dict[str, set[str]]]) -> None:
        self.pools = pools
        self.anchor_calls: list[list[str]] = []
        self.original_chunk_calls: list[tuple[str, int]] = []
        self.rank_calls: list[dict[str, Any]] = []

    def link_entity_ids(self, mentions: list[str], _embedder: object) -> dict[str, list[str]]:
        return {mention: [f"{mention}-id"] for mention in dict.fromkeys(mentions)}

    def candidate_pool(
        self,
        mentions: list[str],
        embedder: object,
        *,
        entity_ids: dict[str, list[str]] | None = None,
    ) -> dict[str, set[str]]:
        assert entity_ids == {mention: [f"{mention}-id"] for mention in dict.fromkeys(mentions)}
        self.anchor_calls.append(list(mentions))
        return self.pools.pop(0)

    def original_question_candidate_pool(
        self,
        question: str,
        _embedder: object,
        *,
        chunk_top_k: int,
    ) -> dict[str, set[str]]:
        self.original_chunk_calls.append((question, chunk_top_k))
        return self.pools.pop(0)

    def rank(
        self,
        question: str,
        candidates: dict[str, set[str]],
        embedder: object,
    ) -> list[dict[str, Any]]:
        self.rank_calls.append({"question": question, "candidates": candidates})
        return [
            {
                "id": hyperedge_id,
                "text": hyperedge_id,
                "chunks": [(hyperedge_id, f"Evidence for {hyperedge_id}")],
                "first_hop_texts": sorted(candidates[hyperedge_id]),
            }
            for hyperedge_id in candidates
        ]


class RecordingClient:
    def __init__(self, entities: list[list[str]], answers: list[str]) -> None:
        self.entities = entities
        self.answers = answers
        self.chat_calls: list[dict[str, Any]] = []
        self.entity_questions: list[str] = []

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)

    def chat_json(
        self, _system_prompt: str, user_prompt: str, *, max_tokens: int | None = None
    ) -> dict[str, object]:
        if max_tokens is None:
            self.entity_questions.append(json.loads(user_prompt)["question"])
            return {"entities": self.entities.pop(0)}
        self.chat_calls.append(json.loads(user_prompt))
        self.assert_max_tokens(max_tokens)
        return {"answer": self.answers.pop(0)}

    @staticmethod
    def assert_max_tokens(max_tokens: int) -> None:
        assert max_tokens == 900
if __name__ == "__main__":
    unittest.main()
