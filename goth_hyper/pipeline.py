from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .data.loaders import HypergraphDatasetLoader
from .llm import LocalHashEmbeddingClient, MockReasoningService, OpenAICompatibleClient, OpenAIReasoningService, PromptManager
from .logging_utils import TraceStore
from .reasoning.controller import ThoughtController
from .reasoning.operations import ThoughtOperationExecutor
from .reasoning.scoring import ThoughtScorer
from .reasoning.taskframe import TaskFrameBuilder, TaskFrameRegistry
from .retrieval.evidence import EvidenceRetriever


class GoTHyperPipeline:
    def __init__(
        self,
        config: Config,
        run_dir: Path,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.trace_store = trace_store

        loader = HypergraphDatasetLoader(config.dataset, logger)
        self.dataset = loader.load()
        self.trace_store.save_artifact("artifacts/dataset_summary.json", self.dataset.summary)

        if config.llm.use_mock:
            self.embedder = LocalHashEmbeddingClient()
            self.llm_service = MockReasoningService()
        else:
            client = OpenAICompatibleClient(config.llm, trace_store=trace_store)
            self.embedder = client
            prompts = PromptManager(config.prompts.directory)
            self.llm_service = OpenAIReasoningService(client=client, prompts=prompts)

        taskframe_builder = TaskFrameBuilder(self.llm_service, self.dataset, logger, trace_store)
        registry = TaskFrameRegistry(
            embedder=self.embedder,
            threshold=config.retrieval.taskframe_registration_threshold,
            logger=logger,
            trace_store=trace_store,
        )
        scorer = ThoughtScorer(embedder=self.embedder, config=config.reasoning, logger=logger)
        evidence_retriever = EvidenceRetriever(
            dataset=self.dataset,
            embedder=self.embedder,
            config=config.retrieval,
            logger=logger,
        )
        executor = ThoughtOperationExecutor(logger=logger, trace_store=trace_store)
        self.controller = ThoughtController(
            config=config,
            dataset=self.dataset,
            taskframe_builder=taskframe_builder,
            registry=registry,
            scorer=scorer,
            evidence_retriever=evidence_retriever,
            executor=executor,
            llm_service=self.llm_service,
            logger=logger,
            trace_store=trace_store,
        )

    def run(self, question: str) -> dict[str, Any]:
        self.logger.info("Starting GoTHyper pipeline for question: %s", question)
        result = self.controller.run(question)
        self.trace_store.save_artifact("artifacts/task_frame.json", result["task_frame"])
        self.trace_store.save_artifact("artifacts/thought_graph.json", result["thought_graph"])
        self.trace_store.save_artifact("artifacts/evidence_subgraph.json", result["evidence_subgraph"])
        self.trace_store.save_artifact("artifacts/final_answer.json", result["final_answer"])
        result["run_dir"] = str(self.run_dir)
        self.logger.info("Pipeline finished. Artifacts saved under %s", self.run_dir)
        return result
