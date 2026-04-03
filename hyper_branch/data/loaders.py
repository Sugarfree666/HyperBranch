from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DatasetConfig
from .graph import KnowledgeHypergraph
from .vector_store import VectorStore


@dataclass(slots=True)
class DatasetBundle:
    root: Path
    graph_path: Path
    graph: KnowledgeHypergraph
    text_chunks: dict[str, dict[str, Any]]
    full_docs: dict[str, dict[str, Any]]
    entity_store: VectorStore
    hyperedge_store: VectorStore
    chunk_store: VectorStore
    summary: dict[str, Any]

    def get_chunk_text(self, chunk_id: str) -> str:
        return str(self.text_chunks.get(chunk_id, {}).get("content", ""))

    def get_chunk_record(self, chunk_id: str) -> dict[str, Any]:
        return dict(self.text_chunks.get(chunk_id, {}))


class HypergraphDatasetLoader:
    def __init__(self, config: DatasetConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def load(self) -> DatasetBundle:
        root = self.config.root
        graph_path = self._resolve_graph_path(root)
        self.logger.info("Loading dataset from %s", root)
        self.logger.info("Using GraphML file %s", graph_path.name)

        text_chunks = self._load_json(root / self.config.text_chunk_file)
        full_docs = self._load_json(root / self.config.full_doc_file)
        entity_vdb_path = root / self.config.entity_vdb_file
        if not entity_vdb_path.exists():
            entity_vdb_path = root / self.config.entity_vdb_fallback_file
        graph = KnowledgeHypergraph.from_graphml(graph_path)
        entity_store = VectorStore.from_json(entity_vdb_path, name="entities", label_fields=("entity_name",))
        hyperedge_store = VectorStore.from_json(
            root / self.config.hyperedge_vdb_file,
            name="hyperedges",
            label_fields=("hyperedge_name",),
        )
        chunk_store = VectorStore.from_json(root / self.config.chunk_vdb_file, name="chunks", label_fields=("__id__",))

        summary = {
            "dataset_root": str(root),
            "graphml_file": graph_path.name,
            "doc_count": len(full_docs),
            "chunk_count": len(text_chunks),
            "entity_vector_count": len(entity_store.rows),
            "hyperedge_vector_count": len(hyperedge_store.rows),
            "chunk_vector_count": len(chunk_store.rows),
            "graph": graph.summarize(),
        }
        return DatasetBundle(
            root=root,
            graph_path=graph_path,
            graph=graph,
            text_chunks=text_chunks,
            full_docs=full_docs,
            entity_store=entity_store,
            hyperedge_store=hyperedge_store,
            chunk_store=chunk_store,
            summary=summary,
        )

    def _resolve_graph_path(self, root: Path) -> Path:
        if self.config.graphml_file:
            explicit = root / self.config.graphml_file
            if explicit.exists():
                return explicit
        preferred = root / "graph_chunk_entity_relation.graphml"
        if preferred.exists():
            return preferred
        graphml_files = sorted(root.glob("*.graphml"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not graphml_files:
            raise FileNotFoundError(f"No GraphML file found under {root}")
        return graphml_files[0]

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
