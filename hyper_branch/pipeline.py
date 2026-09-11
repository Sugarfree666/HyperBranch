"""Run the HyperBranch retrieval pipeline over a DEPO atomic-question DAG."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .client import OpenAIClient
from .database import CandidatePool, HypergraphDatabase

#读取原子问题回答的提示词
ANSWER_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "atomic_answer.md"
).read_text(encoding="utf-8").strip()
ORIGINAL_QUESTION_CHUNK_TOP_K = 5
ANSWER_CHUNK_TOP_K = 10
ENTITY_RECOGNITION_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "entity_recognition.md"
).read_text(encoding="utf-8").strip()


class HyperBranchPipeline:
    """Topologically answer a DEPO DAG using local hypergraph evidence."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        model: str,
        embedding_model: str,
        timeout_seconds: int,
        temperature: float,
        api_key: str,
        base_url: str | None = None,
        client: OpenAIClient | None = None,
        original_question_chunk_top_k: int = ORIGINAL_QUESTION_CHUNK_TOP_K,
    ) -> None:
        self.original_question_chunk_top_k = original_question_chunk_top_k
        #创建超图数据库
        self.database = HypergraphDatabase(dataset_root)
        #创建LLM客户端
        self.client = client or OpenAIClient(
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            base_url=base_url,
        )

    def run(
        self,
        question: str,
        dag: dict[str, Any],
        original_question_entities: list[str],
    ) -> dict[str, Any]:
        #拓扑排序
        nodes = _topological_order(dag.get("nodes", []))
        #创建原始问题的搜索空间
        topic_entity_ids = self.database.link_entity_ids(original_question_entities, self.client)
        original_pool = self.database.original_question_candidate_pool(
            question,
            self.client,
            chunk_top_k=self.original_question_chunk_top_k,
        )
        #保存每个原子问题的答案
        answers: dict[str, dict[str, Any]] = {}
        #每个原子问题自己的局部搜索空间
        local_pools: dict[str, CandidatePool] = {}
        #保存每个原子问题的祖先节点
        ancestors: dict[str, set[str]] = {}
        
        for node in nodes:
            node_id = node["id"]
            #取当前节点的依赖节点id
            dependency_ids = node.get("depends_on", [])
            #取出依赖问题的答案
            dependency_answers = [answers[dependency_id] for dependency_id in dependency_ids]
            dependency_context = _answer_dependency_context(dependency_answers)
            #重写问题
            rewritten_question, _ = _rewrite_question(
                node["question"],
                dependency_answers,
            )
            anchors = self.client.chat_json(
                ENTITY_RECOGNITION_PROMPT,
                json.dumps({"question": rewritten_question}, ensure_ascii=False),
            )["entities"]
            #建立当前原子问题搜索空间
            entity_ids = self.database.link_entity_ids(anchors, self.client)
            local_pool = self.database.candidate_pool(
                anchors,
                self.client,
                entity_ids=entity_ids,
            )
            #记录当前节点的候选池
            local_pools[node_id] = local_pool
            #先将当前的依赖节点加入祖先节点集合
            node_ancestors = set(dependency_ids)
            for dependency_id in dependency_ids:
                #加入依赖节点的祖先节点
                node_ancestors.update(ancestors[dependency_id])
            #更新
            ancestors[node_id] = node_ancestors

            merged_pool: CandidatePool = {}
            active_pools = [original_pool]
            ##加入当前节点的祖先节点候选池
            active_pools.extend(
                local_pools[ancestor["id"]]
                for ancestor in nodes
                if ancestor["id"] in node_ancestors
            )
            
            active_pools.append(local_pool)
            #合并候选池超边
            for pool in active_pools:
                for hyperedge_id, first_hops in pool.items():
                    merged_pool.setdefault(hyperedge_id, set()).update(first_hops)
            
            #返回top-k超边
            ranked_hyperedges = self.database.rank(
                rewritten_question,
                merged_pool,
                self.client,
            )
            evidence_blocks = _evidence_blocks(ranked_hyperedges)
            response = self.client.chat_json(
                ANSWER_PROMPT,
                json.dumps(
                    {
                        "original_question": question,
                        "atomic_question": rewritten_question,
                        "dependency_context": dependency_context,
                        "evidence_blocks": evidence_blocks,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                max_tokens=900,
            )
            answer = str(response["answer"]).strip()
            answers[node_id] = {
                "node_id": node_id,
                "question": rewritten_question,
                "entities": anchors,
                "entity_ids": entity_ids,
                "evidence_blocks": evidence_blocks,
                "answer": answer,
            }

        atomic_answers = [answers[node["id"]] for node in nodes]
        return {
            "question": question,
            "dag": dag,
            "topic_entity_ids": topic_entity_ids,
            "atomic_answers": atomic_answers,
            "final_answer": {"answer": atomic_answers[-1]["answer"]},
        }


def _topological_order(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按依赖关系返回原子问题，使每个节点排在其依赖节点之后。"""
    #创建{"q1":{"id":"q1","depends_on":[]}}，方便快速索引
    by_id = {node["id"]: node for node in nodes}
    #计算入度
    indegree = {node_id: len(node["depends_on"]) for node_id, node in by_id.items()}
    #用来保存当前节点会影响哪些节点
    dependents = {node_id: [] for node_id in by_id}
    #当前节点服务于哪些节点
    for node in nodes:
        for dependency_id in node["depends_on"]:
            dependents[dependency_id].append(node["id"])

    # 没有依赖的节点可以最先处理。
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    order: list[dict[str, Any]] = []
    while ready:
        # 取出一个已满足全部依赖的节点，并释放依赖它的后继节点。
        node_id = ready.pop(0)
        #将入度为0的加入拓扑排序
        order.append(by_id[node_id])
        #找到当前加入拓扑排序的节点的后续节点
        for dependent_id in dependents[node_id]:
            #将它的度数减1
            indegree[dependent_id] -= 1
            #如果后续节点等于0，作为新的入口
            if indegree[dependent_id] == 0:
                ready.append(dependent_id)
    return order


def _answer_dependency_context(
    dependency_answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the prerequisite records exposed to the answer model.

    Entity IDs are retrieval-only artifacts.  Keep the predecessor's question,
    answer, entities, and evidence so downstream answering can verify the
    provenance of a derived value without receiving retrieval implementation
    details.
    """

    return [
        {
            key: value
            for key, value in dependency_answer.items()
            if key != "entity_ids"
        }
        for dependency_answer in dependency_answers
    ]


def _rewrite_question(
    question: str,
    #所有依赖答案
    dependency_context: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    rewritten = question
    #记录哪些依赖答案替换进了当前问题
    inserted_answers: list[str] = []
    for dependency in dependency_context:
        answer = dependency["answer"].strip()
        reference = re.compile(
            rf"\b{re.escape(dependency['node_id'])}(?:\s+answer|['\u2019]s\s+answer)\b",
            flags=re.IGNORECASE,
        )
        #替换后的问题和替换次数
        rewritten, replacements = reference.subn(lambda _match: answer, rewritten)
        if replacements and answer not in inserted_answers:
            inserted_answers.append(answer)
    return rewritten, inserted_answers


def _evidence_blocks(
    hyperedges: list[dict[str, Any]], max_chunks: int = ANSWER_CHUNK_TOP_K
) -> list[dict[str, Any]]:
    """Return the top ranked source chunks, deduplicated by chunk ID."""
    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1")

    blocks: list[dict[str, Any]] = []
    by_chunk_id: dict[str, dict[str, Any]] = {}
    for candidate in hyperedges:
        for chunk_id, text in candidate["chunks"]:
            if chunk_id in by_chunk_id:
                continue
            if len(blocks) >= max_chunks:
                return blocks

            title, separator, body = text.partition("\n")
            block = {
                "chunk_id": f"C{len(blocks) + 1}",
                "title": title.strip() if separator else "",
                "text": body.strip() if separator else title.strip(),
            }
            by_chunk_id[chunk_id] = block
            blocks.append(block)
    return blocks
