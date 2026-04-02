from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..logging_utils import TraceStore
from ..models import ExtractedSubgraph, TaskFrame, VectorMatch


class SubgraphExtractor:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.logger = logger
        self.trace_store = trace_store

    def extract(self, question: str, task_frame: TaskFrame) -> ExtractedSubgraph:
        anchor_queries = task_frame.anchors or [question]
        relation_queries = [task_frame.target] + task_frame.bridges
        relation_queries = [query for query in relation_queries if query] or [question]

        anchor_vectors = self.embedder.embed_texts(anchor_queries, stage="subgraph_entities")
        relation_vectors = self.embedder.embed_texts(relation_queries, stage="subgraph_hyperedges")

        seed_entities = self._aggregate_matches(
            queries=anchor_queries,
            query_vectors=anchor_vectors,
            store=self.dataset.entity_store,
            top_k_per_query=self.config.anchor_top_k_per_anchor,
            keep=self.config.anchor_keep,
        )
        seed_hyperedges = self._aggregate_matches(
            queries=relation_queries,
            query_vectors=relation_vectors,
            store=self.dataset.hyperedge_store,
            top_k_per_query=self.config.relation_top_k_per_query,
            keep=self.config.relation_keep,
        )

        seed_node_ids = [match.label for match in seed_entities + seed_hyperedges]
        subgraph = self.dataset.graph.extract_subgraph(
            seed_node_ids=seed_node_ids,
            hops=self.config.subgraph_hops,
            seed_entities=seed_entities,
            seed_hyperedges=seed_hyperedges,
        )
        self.logger.info(
            "Extracted subgraph with %s nodes, %s edges, %s source chunks",
            len(subgraph.node_ids),
            len(subgraph.edge_ids),
            len(subgraph.source_chunk_ids),
        )
        self.trace_store.log_event(
            "subgraph_extracted",
            {
                "node_count": len(subgraph.node_ids),
                "edge_count": len(subgraph.edge_ids),
                "source_chunk_count": len(subgraph.source_chunk_ids),
            },
        )
        return subgraph

    def _aggregate_matches(
        self,
        queries: list[str],
        query_vectors: list[np.ndarray],
        store: Any,
        top_k_per_query: int,
        keep: int,
    ) -> list[VectorMatch]:
        aggregated: dict[str, VectorMatch] = {}
        query_hits: dict[str, list[str]] = defaultdict(list)
        for query, query_vector in zip(queries, query_vectors, strict=True):
            matches = store.query(query_vector, top_k=top_k_per_query)
            for match in matches:
                existing = aggregated.get(match.label)
                if existing is None or match.score > existing.score:
                    aggregated[match.label] = VectorMatch(
                        item_id=match.item_id,
                        label=match.label,
                        score=match.score,
                        metadata=dict(match.metadata),
                    )
                query_hits[match.label].append(query)

        ranked = sorted(aggregated.values(), key=lambda item: item.score, reverse=True)[:keep]
        for match in ranked:
            match.metadata["matched_queries"] = query_hits.get(match.label, [])
        return ranked
