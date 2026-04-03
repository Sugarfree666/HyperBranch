from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hyper_branch.config import (
    Config,
    DatasetConfig,
    LLMConfig,
    PromptConfig,
    ReasoningConfig,
    RetrievalConfig,
    RuntimeConfig,
)
from hyper_branch.logging_utils import TraceStore
from hyper_branch.models import (
    EvidenceItem,
    GraphNode,
    HyperedgeCandidate,
    RetrievalControlState,
    TaskFrame,
    ThoughtGraph,
    ThoughtState,
    VectorMatch,
)
from hyper_branch.reasoning.controller import ThoughtController
from hyper_branch.reasoning.operations import ThoughtOperationExecutor
from hyper_branch.retrieval.evidence import EvidenceRetriever


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
                branch_result={
                    "recommended_hyperedges": [candidate.to_dict()],
                    "query_texts": ["What concept appears in both settings?", "shared concept"],
                    "control_state": {
                        "iteration": 1,
                        "branch_weights": {"constraint": 0.5, "relation": 0.25, "anchor": 0.25},
                    },
                    "notes": "Constraint operator ranked this hyperedge first.",
                },
                candidates=[candidate],
                evidence_items=[evidence_item],
                parent_ids=["th-root"],
            )

            self.assertEqual(thought.status, "searched")
            self.assertEqual(thought.metadata["branch_kind"], "constraint")
            self.assertEqual(thought.metadata["recommended_count"], 1)
            self.assertEqual(len(thought.metadata["frontier_hyperedges"]), 1)
            self.assertIn("recommended hyperedges", thought.content)
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
    def __init__(self) -> None:
        self.retrieve_calls: list[dict[str, object]] = []

    def anchor_task_frame(self, question: str, task_frame: TaskFrame) -> dict[str, object]:
        del question, task_frame
        candidate = HyperedgeCandidate(
            hyperedge_id='<hyperedge>"Urban farms build community support."',
            hyperedge_text="Urban farms build community support.",
            score=0.92,
            branch_kind="anchor",
            branch_score=0.92,
            coverage_gain=0.8,
            connector_gain=0.7,
            novelty_gain=0.3,
            entity_ids=['"COMMUNITY SUPPORT"'],
            chunk_ids=["chunk-1"],
            support_entities=['"COMMUNITY SUPPORT"'],
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
        control_state: RetrievalControlState,
        evidence_subgraph: dict[str, object] | None = None,
        exclude_hyperedge_ids: set[str] | None = None,
    ) -> list[HyperedgeCandidate]:
        del evidence_subgraph, exclude_hyperedge_ids
        focus = list(control_state.current_focus())
        query_texts = [question, branch_kind, *task_frame.topic_entities, *focus]
        control_state.branch_queries[branch_kind] = query_texts
        control_state.candidate_filters[branch_kind] = {
            "prefer_focus_match": bool(focus),
            "focus_terms": focus,
        }
        self.retrieve_calls.append(
            {
                "iteration": control_state.iteration,
                "branch_kind": branch_kind,
                "focus": focus,
                "weights": dict(control_state.branch_weights),
                "query_texts": list(query_texts),
            }
        )
        suffix = branch_kind.upper()
        focus_bonus = 0.18 if focus and branch_kind == "relation" else 0.0
        constraint_gain = 0.82 if branch_kind == "constraint" else 0.24
        relation_gain = 0.9 if branch_kind == "relation" else 0.28
        connector_gain = 0.78 if branch_kind == "anchor" else 0.36
        coverage_gain = 0.62 if branch_kind == "anchor" else 0.42
        novelty_gain = 0.33 if branch_kind == "anchor" else 0.12
        branch_score = 0.64 + focus_bonus
        return [
            HyperedgeCandidate(
                hyperedge_id=f'<hyperedge>"{suffix} branch supports community support."',
                hyperedge_text=f"{suffix} branch supports community support.",
                score=branch_score,
                branch_kind=branch_kind,
                branch_score=branch_score,
                coverage_gain=coverage_gain,
                constraint_gain=constraint_gain,
                relation_gain=relation_gain,
                connector_gain=connector_gain,
                novelty_gain=novelty_gain,
                focus_gain=focus_bonus,
                entity_ids=['"COMMUNITY SUPPORT"'],
                chunk_ids=[f"chunk-{branch_kind}"],
                support_entities=['"COMMUNITY SUPPORT"'],
                reason=f"{branch_kind} branch ranked this hyperedge via explicit scoring.",
            )
        ]

    def fuse_frontier(
        self,
        task_frame: TaskFrame,
        branch_candidates: dict[str, list[HyperedgeCandidate]],
        evidence_subgraph: dict[str, object],
        control_state: RetrievalControlState,
        top_k: int,
    ) -> tuple[list[HyperedgeCandidate], dict[str, object]]:
        del task_frame
        fused: list[HyperedgeCandidate] = []
        for branch_kind, candidates in branch_candidates.items():
            weight = control_state.branch_weights.get(branch_kind, 0.0)
            for candidate in candidates:
                candidate.fused_score = candidate.branch_score * (1.0 + weight)
                candidate.score = candidate.fused_score
                fused.append(candidate)
        fused.sort(key=lambda item: item.fused_score, reverse=True)
        selected = fused[:top_k]
        preferred_branches = sorted(
            branch_candidates,
            key=lambda kind: max((candidate.branch_score for candidate in branch_candidates[kind]), default=0.0),
            reverse=True,
        )
        merge_result = {
            "frontier_hyperedge_ids": [candidate.hyperedge_id for candidate in selected],
            "frontier": [candidate.to_dict() for candidate in selected],
            "branch_contributions": {
                kind: [candidate.hyperedge_id for candidate in candidates]
                for kind, candidates in branch_candidates.items()
            },
            "preferred_branches": preferred_branches,
            "coverage_summary": {
                "frontier_size": len(selected),
                "evidence_hyperedges": len(evidence_subgraph.get("hyperedge_ids", [])),
                "topic_entity_coverage": 0.75,
            },
            "answer_hypotheses": ["COMMUNITY SUPPORT"],
            "missing_requirements": list(control_state.missing_requirements),
            "next_focus": list(control_state.next_focus),
            "notes": "Fake fused frontier based on branch_score and branch weight.",
        }
        return selected, merge_result

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
    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        merge_result: dict[str, object],
        evidence_subgraph: dict[str, object],
        iteration: int,
        retrieval_control_state: dict[str, object],
    ) -> dict[str, object]:
        del question, task_frame, merge_result, evidence_subgraph, retrieval_control_state
        if iteration == 1:
            return {
                "enough": False,
                "confidence": 0.61,
                "reason": "Need stronger relation closure before final answer synthesis.",
                "missing_requirements": [
                    "Need stronger relation closure between urban farms and community support.",
                ],
                "next_focus": [
                    "relation closure around community support",
                ],
            }
        return {
            "enough": True,
            "confidence": 0.93,
            "reason": "Frontier coverage and relation closure are now sufficient.",
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
            "answer": str(merge_result.get("answer_hypotheses", [""])[0]),
            "reasoning_summary": "Final answer synthesized from fused frontier hyperedges.",
            "confidence": 0.93,
            "remaining_gaps": [],
        }


class ThoughtControllerSelectionTest(unittest.TestCase):
    def test_controller_runs_retrieval_led_loop_and_updates_control_state(self) -> None:
        logger = logging.getLogger("test.controller")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_store = TraceStore(Path(tmp_dir))
            fake_retriever = FakeEvidenceRetriever()
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
                evidence_retriever=fake_retriever,
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
            self.assertGreaterEqual(len(result["evidence_subgraph"]["control_history"]), 3)
            relation_calls = [
                call
                for call in fake_retriever.retrieve_calls
                if call["branch_kind"] == "relation"
            ]
            self.assertEqual(len(relation_calls), 2)
            self.assertEqual(relation_calls[0]["focus"], [])
            self.assertIn("relation closure around community support", relation_calls[1]["focus"])

            latest_control = result["evidence_subgraph"]["control_history"][-1]
            self.assertIn(
                "Need stronger relation closure between urban farms and community support.",
                latest_control["missing_requirements"],
            )
            self.assertIn("relation closure around community support", latest_control["next_focus"])
            self.assertGreater(
                latest_control["branch_weights"]["relation"],
                latest_control["branch_weights"]["constraint"],
            )
            self.assertIn(
                "relation closure around community support",
                latest_control["branch_queries"]["relation"],
            )


if __name__ == "__main__":
    unittest.main()

