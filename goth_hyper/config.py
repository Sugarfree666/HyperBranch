from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    root: Path
    graphml_file: str | None = None
    full_doc_file: str = "kv_store_full_docs.json"
    text_chunk_file: str = "kv_store_text_chunks.json"
    hyperedge_vdb_file: str = "vdb_hyperedges.json"
    entity_vdb_file: str = "vdb_entity_names.json"
    entity_vdb_fallback_file: str = "vdb_entities.json"
    chunk_vdb_file: str = "vdb_chunks.json"


@dataclass(slots=True)
class RuntimeConfig:
    base_run_dir: Path
    log_level: str = "INFO"


@dataclass(slots=True)
class RetrievalConfig:
    entity_top_k: int = 6
    hyperedge_top_k: int = 6
    chunk_top_k: int = 8
    evidence_keep: int = 6
    taskframe_registration_threshold: float = 0.33


@dataclass(slots=True)
class ReasoningConfig:
    max_steps: int = 5
    coarse_top_k: int = 6
    llm_top_k: int = 3
    thought_score_threshold: float = 0.12
    min_verified_evidence: int = 2
    evidence_score_threshold: float = 0.2


@dataclass(slots=True)
class LLMConfig:
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    timeout_seconds: int = 120
    temperature: float = 0.2
    use_mock: bool = False


@dataclass(slots=True)
class PromptConfig:
    directory: Path


@dataclass(slots=True)
class Config:
    project_root: Path
    dataset: DatasetConfig
    runtime: RuntimeConfig
    retrieval: RetrievalConfig
    reasoning: ReasoningConfig
    llm: LLMConfig
    prompts: PromptConfig


def load_config(config_path: Path, project_root: Path) -> Config:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = raw.get("dataset", {})
    runtime = raw.get("runtime", {})
    retrieval = raw.get("retrieval", {})
    reasoning = raw.get("reasoning", {})
    llm = raw.get("llm", {})
    prompts = raw.get("prompts", {})

    dataset_cfg = DatasetConfig(
        root=_resolve_path(project_root, dataset.get("root", "datasets/agriculture")),
        graphml_file=dataset.get("graphml_file"),
        full_doc_file=dataset.get("full_doc_file", "kv_store_full_docs.json"),
        text_chunk_file=dataset.get("text_chunk_file", "kv_store_text_chunks.json"),
        hyperedge_vdb_file=dataset.get("hyperedge_vdb_file", "vdb_hyperedges.json"),
        entity_vdb_file=dataset.get("entity_vdb_file", "vdb_entity_names.json"),
        entity_vdb_fallback_file=dataset.get("entity_vdb_fallback_file", "vdb_entities.json"),
        chunk_vdb_file=dataset.get("chunk_vdb_file", "vdb_chunks.json"),
    )
    runtime_cfg = RuntimeConfig(
        base_run_dir=_resolve_path(project_root, runtime.get("base_run_dir", "runs")),
        log_level=str(runtime.get("log_level", "INFO")).upper(),
    )
    retrieval_cfg = RetrievalConfig(
        entity_top_k=int(retrieval.get("entity_top_k", 6)),
        hyperedge_top_k=int(retrieval.get("hyperedge_top_k", 6)),
        chunk_top_k=int(retrieval.get("chunk_top_k", 8)),
        evidence_keep=int(retrieval.get("evidence_keep", 6)),
        taskframe_registration_threshold=float(retrieval.get("taskframe_registration_threshold", 0.33)),
    )
    reasoning_cfg = ReasoningConfig(
        max_steps=int(reasoning.get("max_steps", 5)),
        coarse_top_k=int(reasoning.get("coarse_top_k", 6)),
        llm_top_k=int(reasoning.get("llm_top_k", 3)),
        thought_score_threshold=float(reasoning.get("thought_score_threshold", 0.12)),
        min_verified_evidence=int(reasoning.get("min_verified_evidence", 2)),
        evidence_score_threshold=float(reasoning.get("evidence_score_threshold", 0.2)),
    )
    llm_cfg = LLMConfig(
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        base_url_env=str(llm.get("base_url_env", "OPENAI_BASE_URL")),
        model=str(llm.get("model", "gpt-4o-mini")),
        embedding_model=str(llm.get("embedding_model", "text-embedding-3-small")),
        timeout_seconds=int(llm.get("timeout_seconds", 120)),
        temperature=float(llm.get("temperature", 0.2)),
        use_mock=bool(llm.get("use_mock", False)),
    )
    prompt_cfg = PromptConfig(directory=_resolve_path(project_root, prompts.get("dir", "prompts")))
    return Config(
        project_root=project_root,
        dataset=dataset_cfg,
        runtime=runtime_cfg,
        retrieval=retrieval_cfg,
        reasoning=reasoning_cfg,
        llm=llm_cfg,
        prompts=prompt_cfg,
    )


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()
