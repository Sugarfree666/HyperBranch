from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from goth_hyper.config import (
    Config,
    DatasetConfig,
    LLMConfig,
    PromptConfig,
    ReasoningConfig,
    RetrievalConfig,
    RuntimeConfig,
)
from goth_hyper.logging_utils import TraceStore
from goth_hyper.models import EvidenceItem, GraphNode, HyperedgeCandidate, TaskFrame, ThoughtGraph, ThoughtState, VectorMatch
from goth_hyper.reasoning.controller import ThoughtController
from goth_hyper.reasoning.operations import ThoughtOperationExecutor
from goth_hyper.retrieval.evidence import EvidenceRetriever


class ThoughtOperationExecutorTest(unittest.TestCase):
    def test_create_branch_thought_records_branch_metadata_and_evidence(self) -> None:
        logger = logging.getLogger("test.executor")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = ThoughtOperationExecutor(logger=logger, trace_store=TraceStore(Path(tmp_dir)))
            task_frame = TaskFrame.from_payload(
                "What concept appears in both settings?",
                {
                    "topic_entities": ["ecological horticulture apprenticeships", "summer programs"],
                    "answer_type_hint": "concept",
                    "relation_intent": "shared concept",
                    "hard_constraints": ["The answer must be present in both settings."],
                },
            )
            candidate = HyperedgeCandidate(
                hyperedge_id='<hyperedge>"Responsibility is taught through youth farming programs."',
                hyperedge_text="Responsibility is taught through youth farming programs.",
                score=0.88,
                branch_kind="constraint",
                entity_ids=['"RESPONSIBILITY"'],
                chunk_ids=["chunk-1"],
            )
            evidence_item = EvidenceItem(
                evidence_id="ev-th-0001-1",
                chunk_id="chunk-1",
                content="Responsibility is taught through youth farming programs.",
                score=0.88,
                source_node_ids=[candidate.hyperedge_id, '"RESPONSIBILITY"'],
            )

            thought = executor.create_branch_thought(
                thought_id="th-0001",
                task_frame=task_frame,
                branch_kind="constraint",
                iteration=1,
                selection_payload={
                    "candidate_answer": "RESPONSIBILITY",
                    "confidence": 0.91,
                    "supporting_facts": ["Responsibility appears in youth farming programs."],
                    "missing_requirements": [],
                    "notes": "Strong overlap with the question constraints.",
                },
                candidates=[candidate],
                evidence_items=[evidence_item],
                parent_ids=["th-root"],
            )

            self.assertEqual(thought.status, "verified")
            self.assertEqual(thought.metadata["branch_kind"], "constraint")
            self.assertEqual(thought.metadata["candidate_answer"], "RESPONSIBILITY")
            self.assertEqual(thought.grounding.chunk_ids, ["chunk-1"])
            self.assertEqual(len(thought.grounding.evidence), 1)


class CountingStore:
    def __init__(self, matches: list[VectorMatch]) -> None:
        self.matches = matches
        self.query_calls = 0

    def query(self, vector: object, top_k: int) -> list[VectorMatch]:
        del vector
        self.query_calls += 1
        return self.matches[:top_k]


class EmbedderStub:
    def embed_texts(self, texts: list[str], stage: str) -> list[list[float]]:
        del stage
        return [[float(index + 1)] for index, _ in enumerate(texts)]


class FakeGraph:
    def __init__(self) -> None:
        self.nodes = {
            '"COMMUNITY SUPPORT"': GraphNode(node_id='"COMMUNITY SUPPORT"', role="entity", source_ids=["chunk-1"]),
            '<hyperedge>"Urban farms build community support."': GraphNode(
                node_id='<hyperedge>"Urban farms build community support."',
                role="hyperedge",
                source_ids=["chunk-1"],
            ),
        }
        self.adjacency = {'<hyperedge>"Urban farms build community support."': ["edge-1"]}

    def expand_from_entities(self, entity_ids: list[str]) -> list[str]:
        if '"COMMUNITY SUPPORT"' in entity_ids:
            return ['<hyperedge>"Urban farms build community support."']
        return []

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        if hyperedge_id == '<hyperedge>"Urban farms build community support."':
            return ['"COMMUNITY SUPPORT"']
        return []

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        if hyperedge_id == '<hyperedge>"Urban farms build community support."':
            return ["chunk-1"]
        return []


class EvidenceRetrieverCacheTest(unittest.TestCase):
    def test_retrieve_reuses_cached_results_for_unchanged_thought(self) -> None:
        logger = logging.getLogger("test.retriever")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        chunk_store = CountingStore([VectorMatch(item_id="chunk-1", label="chunk-1", score=0.95)])
        entity_store = CountingStore([VectorMatch(item_id="entity-1", label='"COMMUNITY SUPPORT"', score=0.81)])
        hyperedge_store = CountingStore(
            [
                VectorMatch(
                    item_id="rel-1",
                    label='<hyperedge>"Urban farms build community support."',
                    score=0.91,
                )
            ]
        )
        dataset = SimpleNamespace(
            chunk_store=chunk_store,
            entity_store=entity_store,
            hyperedge_store=hyperedge_store,
            graph=FakeGraph(),
            get_chunk_text=lambda chunk_id: "Urban farms build community support." if chunk_id == "chunk-1" else "",
        )
        retriever = EvidenceRetriever(
            dataset=dataset,
            embedder=EmbedderStub(),
            config=RetrievalConfig(entity_top_k=1, hyperedge_top_k=1, chunk_top_k=1, evidence_keep=1),
            logger=logger,
        )
        thought = ThoughtState(
            thought_id="th-0007",
            kind="reasoning",
            content="Urban farms build community support",
            objective="Find the supporting concept",
            slot_id="target-0",
            grounding=SimpleNamespace(anchor_texts=["urban farms", "community support"], node_ids=[], chunk_ids=[]),
            metadata={"branch_kind": "anchor"},
        )

        first_pass = retriever.retrieve(thought)
        first_counts = (chunk_store.query_calls, entity_store.query_calls, hyperedge_store.query_calls)
        second_pass = retriever.retrieve(thought)
        second_counts = (chunk_store.query_calls, entity_store.query_calls, hyperedge_store.query_calls)

        self.assertEqual(len(first_pass), 1)
        self.assertEqual(len(second_pass), 1)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first_pass[0].evidence_id, second_pass[0].evidence_id)


class FakeTaskFrameBuilder:
    def build(self, question: str) -> TaskFrame:
        return TaskFrame.from_payload(
            question,
            {
                "topic_entities": ["urban farms", "community support"],
                "answer_type_hint": "concept",
                "relation_intent": "find the shared concept",
                "hard_constraints": ["Ground the answer in retrieved evidence."],
            },
        )


class FakeRegistry:
    def register_anchor_matches(self, task_frame: TaskFrame, entity_matches: list[VectorMatch]) -> list[str]:
        del entity_matches
        if task_frame.anchors:
            task_frame.mark_slot("anchor-0", status="retrieved", note="Registered anchor")
            return ["anchor-0"]
        return []

    def register_reasoning(self, task_frame: TaskFrame, reasoning_thought: ThoughtState) -> list[str]:
        del reasoning_thought
        return ["target-0"]


class FakeEvidenceRetriever:
    def anchor_task_frame(self, question: str, task_frame: TaskFrame) -> dict[str, object]:
        del question, task_frame
        candidate = HyperedgeCandidate(
            hyperedge_id='<hyperedge>"Urban farms build community support."',
            hyperedge_text="Urban farms build community support.",
            score=0.92,
            branch_kind="anchor",
            entity_ids=['"COMMUNITY SUPPORT"'],
            chunk_ids=["chunk-1"],
        )
        return {
            "entity_matches": [VectorMatch(item_id="entity-1", label='"COMMUNITY SUPPORT"', score=0.81)],
            "initial_entity_ids": ['"COMMUNITY SUPPORT"'],
            "initial_hyperedge_ids": [candidate.hyperedge_id],
            "initial_hyperedge_candidates": [candidate],
        }

    def retrieve_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        evidence_subgraph: dict[str, object] | None = None,
        exclude_hyperedge_ids: set[str] | None = None,
    ) -> list[HyperedgeCandidate]:
        del question, task_frame, evidence_subgraph, exclude_hyperedge_ids
        suffix = branch_kind.upper()
        return [
            HyperedgeCandidate(
                hyperedge_id=f'<hyperedge>"{suffix} branch supports community support."',
                hyperedge_text=f"{suffix} branch supports community support.",
                score=0.8,
                branch_kind=branch_kind,
                entity_ids=['"COMMUNITY SUPPORT"'],
                chunk_ids=[f"chunk-{branch_kind}"],
            )
        ]

    def build_evidence_items(
        self,
        thought_id: str,
        branch_kind: str,
        candidates: list[HyperedgeCandidate],
        limit: int | None = None,
    ) -> list[EvidenceItem]:
        del limit
        if not candidates:
            return []
        candidate = candidates[0]
        return [
            EvidenceItem(
                evidence_id=f"ev-{thought_id}-1",
                chunk_id=candidate.chunk_ids[0],
                content=candidate.hyperedge_text,
                score=candidate.score,
                source_node_ids=[candidate.hyperedge_id, *candidate.entity_ids],
            )
        ]


class FakeLLMService:
    def select_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        candidate_hyperedges: list[HyperedgeCandidate],
        evidence_subgraph: dict[str, object],
        top_k: int,
    ) -> dict[str, object]:
        del question, task_frame, evidence_subgraph, top_k
        candidate = candidate_hyperedges[0]
        return {
            "selected_hyperedge_ids": [candidate.hyperedge_id],
            "candidate_answer": "COMMUNITY SUPPORT",
            "supporting_facts": [f"{branch_kind} branch found support evidence."],
            "missing_requirements": [],
            "confidence": 0.9,
            "notes": "Stub branch selection.",
        }

    def reconcile_branches(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, object]],
        evidence_subgraph: dict[str, object],
    ) -> dict[str, object]:
        del question, task_frame, evidence_subgraph
        return {
            "consensus_answer": "COMMUNITY SUPPORT",
            "agreement_groups": [[record["branch_kind"] for record in branch_results]],
            "conflicts": [],
            "preferred_branches": [record["branch_kind"] for record in branch_results],
            "missing_requirements": [],
            "notes": "All three branches agree.",
        }

    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, object]],
        merge_result: dict[str, object],
        evidence_subgraph: dict[str, object],
        iteration: int,
    ) -> dict[str, object]:
        del question, task_frame, branch_results, evidence_subgraph, iteration
        return {
            "enough": bool(merge_result.get("consensus_answer")),
            "confidence": 0.93,
            "reason": "Consensus answer already grounded by all branches.",
            "missing_requirements": [],
            "next_focus": [],
        }

    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        evidence_subgraph: dict[str, object],
        merge_result: dict[str, object],
    ) -> dict[str, object]:
        del question, task_frame, thought_graph, evidence_subgraph
        return {
            "answer": str(merge_result.get("consensus_answer", "")),
            "reasoning_summary": "Three branches converged on the same grounded answer.",
            "confidence": 0.93,
            "remaining_gaps": [],
        }


class ThoughtControllerSelectionTest(unittest.TestCase):
    def test_controller_runs_three_branches_and_builds_evidence_subgraph(self) -> None:
        logger = logging.getLogger("test.controller")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_store = TraceStore(Path(tmp_dir))
            config = Config(
                project_root=Path(tmp_dir),
                dataset=DatasetConfig(root=Path(tmp_dir)),
                runtime=RuntimeConfig(base_run_dir=Path(tmp_dir)),
                retrieval=RetrievalConfig(),
                reasoning=ReasoningConfig(max_steps=2, branch_top_k=1, evidence_top_k_per_branch=1),
                llm=LLMConfig(use_mock=True),
                prompts=PromptConfig(directory=Path(tmp_dir)),
            )
            controller = ThoughtController(
                config=config,
                dataset=SimpleNamespace(),
                taskframe_builder=FakeTaskFrameBuilder(),
                registry=FakeRegistry(),
                scorer=SimpleNamespace(),
                evidence_retriever=FakeEvidenceRetriever(),
                executor=ThoughtOperationExecutor(logger=logger, trace_store=trace_store),
                llm_service=FakeLLMService(),
                logger=logger,
                trace_store=trace_store,
            )

            result = controller.run("How can urban farms build community support?")

            self.assertEqual(result["final_answer"]["answer"], "COMMUNITY SUPPORT")
            self.assertIn("constraint", result["evidence_subgraph"]["branch_support"])
            self.assertIn("relation", result["evidence_subgraph"]["branch_support"])
            self.assertIn("anchor", result["evidence_subgraph"]["branch_support"])
            self.assertTrue(result["thought_graph"]["final_answer"]["answer"])


if __name__ == "__main__":
    unittest.main()
