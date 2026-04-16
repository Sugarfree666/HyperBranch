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
    EvidenceSubgraph,
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
        *,
        channel_id: str = "",
    ) -> list[HyperedgeCandidate]:
        del evidence_subgraph, exclude_hyperedge_ids
        focus = list(control_state.current_focus())
        query_texts = [question, branch_kind, *task_frame.topic_entities, *focus]
        control_state.branch_queries[f"{channel_id}::{branch_kind}"] = query_texts
        control_state.branch_queries[branch_kind] = query_texts
        control_state.candidate_filters[branch_kind] = {
            "prefer_focus_match": bool(focus),
            "focus_terms": focus,
        }
        self.retrieve_calls.append(
            {
                "iteration": control_state.iteration,
                "branch_kind": branch_kind,
                "channel_id": channel_id,
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
        anchor_entity = channel_id or '"COMMUNITY SUPPORT"'
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
                entity_ids=[anchor_entity],
                chunk_ids=[f"chunk-{branch_kind}"],
                support_entities=[anchor_entity],
                channel_id=channel_id,
                supporting_channel_ids=[channel_id] if channel_id else [],
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

    def combine_channel_frontiers(
        self,
        task_frame: TaskFrame,
        channel_frontiers: dict[str, list[HyperedgeCandidate]],
        channel_merge_results: dict[str, dict[str, object]],
        evidence_subgraph: dict[str, object],
        control_state: RetrievalControlState,
        top_k: int,
    ) -> tuple[list[HyperedgeCandidate], dict[str, object]]:
        del task_frame, evidence_subgraph
        fused: list[HyperedgeCandidate] = []
        branch_contributions: dict[str, list[str]] = {}
        for channel_id, candidates in channel_frontiers.items():
            preferred_branches = list(channel_merge_results.get(channel_id, {}).get("preferred_branches", []))
            for branch_kind in preferred_branches:
                branch_contributions.setdefault(str(branch_kind), [])
            for candidate in candidates:
                candidate.supporting_channel_ids = [channel_id]
                candidate.fused_score = candidate.score
                fused.append(candidate)
                for branch_kind in preferred_branches:
                    branch_contributions[str(branch_kind)].append(candidate.hyperedge_id)
        fused.sort(key=lambda item: item.fused_score, reverse=True)
        selected = fused[:top_k]
        return selected, {
            "frontier_hyperedge_ids": [candidate.hyperedge_id for candidate in selected],
            "frontier": [candidate.to_dict() for candidate in selected],
            "branch_contributions": branch_contributions,
            "channel_frontiers": {
                channel_id: [candidate.hyperedge_id for candidate in candidates]
                for channel_id, candidates in channel_frontiers.items()
            },
            "preferred_branches": ["relation", "constraint", "anchor"],
            "coverage_summary": {
                "frontier_size": len(selected),
                "evidence_hyperedges": 0,
                "topic_entity_coverage": 0.75,
                "active_channels": len(channel_frontiers),
            },
            "answer_hypotheses": ["COMMUNITY SUPPORT"],
            "missing_requirements": list(control_state.missing_requirements),
            "next_focus": list(control_state.next_focus),
            "notes": "Fake global frontier aggregated from entity channels.",
        }

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

    def rank_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        frontier_candidates: list[HyperedgeCandidate],
        control_state: RetrievalControlState,
        *,
        exclude_entity_ids: set[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        del question, task_frame, frontier_candidates, control_state, exclude_entity_ids
        return [
            {
                "entity_id": '"URBAN FARM NETWORK"',
                "entity_label": "URBAN FARM NETWORK",
                "description": "A fresh entity for controlled frontier expansion.",
                "source_hyperedges": ["ANCHOR branch supports community support."],
                "question_match": 0.63,
                "focus_match": 0.21,
                "support_count": 1,
                "coarse_score": 0.61,
            },
            {
                "entity_id": '"LOCAL COOPERATIVE"',
                "entity_label": "LOCAL COOPERATIVE",
                "description": "A secondary expansion entity for controlled frontier expansion.",
                "source_hyperedges": ["RELATION branch supports community support."],
                "question_match": 0.47,
                "focus_match": 0.12,
                "support_count": 1,
                "coarse_score": 0.45,
            },
        ][:top_k]


class FakeLLMService:
    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        llm_evidence_view: dict[str, object],
        iteration: int,
    ) -> dict[str, object]:
        del question, task_frame, llm_evidence_view
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
        llm_evidence_view: dict[str, object],
    ) -> dict[str, object]:
        del question, task_frame, thought_graph
        return {
            "answer": str(llm_evidence_view.get("coverage_summary", {}).get("answer_hypotheses", [""])[0]),
            "reasoning_summary": "Final answer synthesized from fused frontier hyperedges.",
            "confidence": 0.93,
            "remaining_gaps": [],
        }

    def select_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_entities: list[dict[str, object]],
        control_state: RetrievalControlState,
    ) -> dict[str, object]:
        del question, task_frame, control_state
        return {
            "selected_entity_ids": [
                str(candidate["entity_id"])
                for candidate in candidate_entities[:2]
            ],
            "reason": "Fake LLM reranked fresh frontier entities.",
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
            self.assertEqual(
                result["evidence_subgraph"]["expansion_frontier_entity_ids"],
                ['"URBAN FARM NETWORK"', '"LOCAL COOPERATIVE"'],
            )

    def test_controller_runs_parallel_entity_channels(self) -> None:
        logger = logging.getLogger("test.controller.parallel")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        class MultiChannelEvidenceRetriever(FakeEvidenceRetriever):
            def anchor_task_frame(self, question: str, task_frame: TaskFrame) -> dict[str, object]:
                del question, task_frame
                first = HyperedgeCandidate(
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
                second = HyperedgeCandidate(
                    hyperedge_id='<hyperedge>"Urban farms partner with local cooperatives."',
                    hyperedge_text="Urban farms partner with local cooperatives.",
                    score=0.87,
                    branch_kind="anchor",
                    branch_score=0.87,
                    coverage_gain=0.74,
                    connector_gain=0.66,
                    novelty_gain=0.28,
                    entity_ids=['"LOCAL COOPERATIVE"'],
                    chunk_ids=["chunk-2"],
                    support_entities=['"LOCAL COOPERATIVE"'],
                )
                return {
                    "entity_matches": [
                        VectorMatch(item_id="entity-1", label='"COMMUNITY SUPPORT"', score=0.81),
                        VectorMatch(item_id="entity-2", label='"LOCAL COOPERATIVE"', score=0.79),
                    ],
                    "initial_entity_ids": ['"COMMUNITY SUPPORT"', '"LOCAL COOPERATIVE"'],
                    "initial_hyperedge_ids": [first.hyperedge_id, second.hyperedge_id],
                    "initial_hyperedge_candidates": [first, second],
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_store = TraceStore(Path(tmp_dir))
            fake_retriever = MultiChannelEvidenceRetriever()
            config = Config(
                project_root=Path(tmp_dir),
                dataset=DatasetConfig(root=Path(tmp_dir)),
                runtime=RuntimeConfig(base_run_dir=Path(tmp_dir)),
                retrieval=RetrievalConfig(),
                reasoning=ReasoningConfig(max_steps=1, branch_top_k=1, evidence_top_k_per_branch=1),
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

            channel_ids = set(result["evidence_subgraph"]["active_channel_ids"])
            self.assertEqual(channel_ids, {'"COMMUNITY SUPPORT"', '"LOCAL COOPERATIVE"'})
            retrieved_channels = {str(call["channel_id"]) for call in fake_retriever.retrieve_calls}
            self.assertEqual(retrieved_channels, {'"COMMUNITY SUPPORT"', '"LOCAL COOPERATIVE"'})
            self.assertIn("entity_channels", result["evidence_subgraph"])
            self.assertEqual(len(result["evidence_subgraph"]["entity_channels"]), 2)


class EmptyStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.row_ids: list[str] = []

    def query(self, vector: object, top_k: int) -> list[VectorMatch]:
        del vector, top_k
        return []

    def similarity(self, query_vector: object, row_id: str) -> float:
        del query_vector, row_id
        return 0.0

    def _label_for_row(self, row: dict[str, object], fallback: str) -> str:
        del row
        return fallback


class ControlledFrontierGraph:
    def __init__(self) -> None:
        self.nodes = {
            '"ENTITY A"': GraphNode(
                node_id='"ENTITY A"',
                role="entity",
                description="Previously explored entity A.",
                source_ids=["chunk-a"],
            ),
            '"ENTITY B"': GraphNode(
                node_id='"ENTITY B"',
                role="entity",
                description="Fresh entity about lead contamination and urban farms.",
                source_ids=["chunk-b"],
            ),
            '"ENTITY C"': GraphNode(
                node_id='"ENTITY C"',
                role="entity",
                description="Irrelevant entity about goat milking routines.",
                source_ids=["chunk-c"],
            ),
            '<hyperedge>"A links to B"': GraphNode(
                node_id='<hyperedge>"A links to B"',
                role="hyperedge",
                source_ids=["chunk-ab"],
            ),
            '<hyperedge>"A only"': GraphNode(
                node_id='<hyperedge>"A only"',
                role="hyperedge",
                source_ids=["chunk-a-only"],
            ),
            '<hyperedge>"B only"': GraphNode(
                node_id='<hyperedge>"B only"',
                role="hyperedge",
                source_ids=["chunk-b-only"],
            ),
        }
        self.adjacency = {
            '<hyperedge>"A links to B"': [],
            '<hyperedge>"A only"': [],
            '<hyperedge>"B only"': [],
        }

    def expand_from_entities(self, entity_ids: list[str]) -> list[str]:
        expanded: list[str] = []
        for entity_id in entity_ids:
            if entity_id == '"ENTITY A"':
                expanded.extend(['<hyperedge>"A links to B"', '<hyperedge>"A only"'])
            if entity_id == '"ENTITY B"':
                expanded.extend(['<hyperedge>"A links to B"', '<hyperedge>"B only"'])
        return expanded

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        mapping = {
            '<hyperedge>"A links to B"': ['"ENTITY A"', '"ENTITY B"'],
            '<hyperedge>"A only"': ['"ENTITY A"'],
            '<hyperedge>"B only"': ['"ENTITY B"', '"ENTITY C"'],
        }
        return mapping.get(hyperedge_id, [])

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        mapping = {
            '<hyperedge>"A links to B"': ["chunk-ab"],
            '<hyperedge>"A only"': ["chunk-a-only"],
            '<hyperedge>"B only"': ["chunk-b-only"],
        }
        return mapping.get(hyperedge_id, [])


class ControlledExpansionTest(unittest.TestCase):
    def test_evidence_subgraph_does_not_promote_source_node_ids_into_expansion_entities(self) -> None:
        subgraph = EvidenceSubgraph()
        subgraph.seed_expansion_frontier(['"ENTITY A"'])
        subgraph.add_frontier(
            iteration=1,
            candidates=[
                HyperedgeCandidate(
                    hyperedge_id='<hyperedge>"A links to B"',
                    hyperedge_text="A links to B",
                    score=0.8,
                    branch_kind="frontier",
                    entity_ids=['"ENTITY B"'],
                    chunk_ids=["chunk-ab"],
                )
            ],
            evidence_items=[
                EvidenceItem(
                    evidence_id="ev-1",
                    chunk_id="chunk-ab",
                    content="A links to B",
                    score=0.8,
                    source_node_ids=['<hyperedge>"A links to B"', '"IRRELEVANT SOURCE ENTITY"'],
                )
            ],
            control_state={"iteration": 1},
            expansion_state={
                "selected_entity_ids": ['"ENTITY B"'],
                "explored_entity_ids": ['"ENTITY A"'],
                "candidate_entities": [],
                "reason": "test",
            },
        )

        self.assertNotIn('"IRRELEVANT SOURCE ENTITY"', subgraph.entity_ids)
        self.assertEqual(subgraph.expansion_frontier_entity_ids, ['"ENTITY B"'])
        self.assertEqual(subgraph.explored_entity_ids, ['"ENTITY A"'])

    def test_retriever_only_expands_from_controlled_frontier_entities(self) -> None:
        logger = logging.getLogger("test.controlled_frontier")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        dataset = SimpleNamespace(
            chunk_store=EmptyStore(),
            entity_store=EmptyStore(),
            hyperedge_store=EmptyStore(),
            graph=ControlledFrontierGraph(),
            get_chunk_text=lambda chunk_id: {
                "chunk-ab": "Entity B is the next useful frontier for lead contamination work.",
                "chunk-a-only": "Entity A stays on the already explored branch.",
                "chunk-b-only": "Entity B connects to lead contamination and urban farms.",
                "chunk-b": "Entity B description support.",
                "chunk-c": "Entity C is mostly about goats.",
            }.get(chunk_id, ""),
        )
        retriever = EvidenceRetriever(
            dataset=dataset,
            embedder=EmbedderStub(),
            config=RetrievalConfig(entity_top_k=1, hyperedge_top_k=3, chunk_top_k=1, evidence_keep=1, branch_candidate_pool=5),
            logger=logger,
        )
        task_frame = TaskFrame.from_payload(
            "How should urban farms handle lead contamination next?",
            {
                "topic_entities": ["urban farms", "lead contamination"],
                "answer_type_hint": "entity",
                "relation_intent": "find the next useful frontier entity",
                "hard_constraints": ["Prefer the next unexplored entity."],
            },
        )
        task_frame.initial_entity_ids = ['"ENTITY A"']
        control_state = RetrievalControlState(iteration=2)

        candidates = retriever.retrieve_branch_candidates(
            question=task_frame.question,
            task_frame=task_frame,
            branch_kind="anchor",
            control_state=control_state,
            evidence_subgraph={
                "entity_ids": ['"ENTITY A"', '"ENTITY B"', '"ENTITY C"'],
                "hyperedge_ids": ['<hyperedge>"A links to B"'],
                "expansion_frontier_entity_ids": ['"ENTITY B"'],
                "explored_entity_ids": ['"ENTITY A"'],
            },
        )
        expansion_entities = retriever.rank_expansion_entities(
            question=task_frame.question,
            task_frame=task_frame,
            frontier_candidates=[
                HyperedgeCandidate(
                    hyperedge_id='<hyperedge>"B only"',
                    hyperedge_text="B only",
                    score=0.5,
                    branch_kind="frontier",
                    entity_ids=['"ENTITY B"', '"ENTITY C"'],
                    chunk_ids=["chunk-b-only"],
                )
            ],
            control_state=control_state,
            exclude_entity_ids={'"ENTITY A"', '"ENTITY B"'},
            top_k=2,
        )
        ranked_without_current_frontier_exclusion = retriever.rank_expansion_entities(
            question=task_frame.question,
            task_frame=task_frame,
            frontier_candidates=[
                HyperedgeCandidate(
                    hyperedge_id='<hyperedge>"B only"',
                    hyperedge_text="B only",
                    score=0.5,
                    branch_kind="frontier",
                    entity_ids=['"ENTITY B"', '"ENTITY C"'],
                    chunk_ids=["chunk-b-only"],
                )
            ],
            control_state=control_state,
            exclude_entity_ids={'"ENTITY A"'},
            top_k=2,
        )

        candidate_ids = [candidate.hyperedge_id for candidate in candidates]
        self.assertIn('<hyperedge>"B only"', candidate_ids)
        self.assertNotIn('<hyperedge>"A only"', candidate_ids)
        self.assertEqual(expansion_entities[0]["entity_id"], '"ENTITY C"')
        self.assertEqual(ranked_without_current_frontier_exclusion[0]["entity_id"], '"ENTITY B"')


if __name__ == "__main__":
    unittest.main()

