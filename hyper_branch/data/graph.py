from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from ..models import GraphEdge, GraphNode
from ..utils import split_source_ids


class KnowledgeHypergraph:
    def __init__(self, nodes: dict[str, GraphNode], edges: dict[str, GraphEdge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.source_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.source_to_edges: dict[str, list[str]] = defaultdict(list)

        for edge_id, edge in edges.items():
            self.adjacency[edge.source].append(edge_id)
            self.adjacency[edge.target].append(edge_id)
            for source_id in edge.source_ids:
                self.source_to_edges[source_id].append(edge_id)

        for node_id, node in nodes.items():
            for source_id in node.source_ids:
                self.source_to_nodes[source_id].append(node_id)

    @classmethod
    def from_graphml(cls, path: Path) -> "KnowledgeHypergraph":
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        tree = ET.parse(path)
        root = tree.getroot()
        key_names = {
            key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"])
            for key in root.findall("g:key", namespace)
        }
        graph = root.find("g:graph", namespace)
        if graph is None:
            raise ValueError(f"Could not find GraphML graph element in {path}.")

        nodes: dict[str, GraphNode] = {}
        for node_elem in graph.findall("g:node", namespace):
            node_id = node_elem.attrib["id"]
            data = _collect_graphml_data(node_elem, key_names, namespace)
            nodes[node_id] = GraphNode(
                node_id=node_id,
                role=data.get("role", ""),
                weight=float(data.get("weight", 0.0) or 0.0),
                source_ids=split_source_ids(data.get("source_id", "")),
                entity_type=data.get("entity_type"),
                description=data.get("description"),
            )

        edges: dict[str, GraphEdge] = {}
        for index, edge_elem in enumerate(graph.findall("g:edge", namespace)):
            data = _collect_graphml_data(edge_elem, key_names, namespace)
            edge_id = f"edge-{index}"
            weight = data.get("weight")
            if weight is None:
                weight = data.get("weight_float")
            edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                source=edge_elem.attrib["source"],
                target=edge_elem.attrib["target"],
                role=data.get("role", ""),
                weight=float(weight or 0.0),
                source_ids=split_source_ids(data.get("source_id", "")),
            )
        return cls(nodes=nodes, edges=edges)

    def summarize(self) -> dict[str, Any]:
        role_counts = Counter(node.role for node in self.nodes.values())
        edge_role_counts = Counter(edge.role for edge in self.edges.values())
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_roles": dict(role_counts),
            "edge_roles": dict(edge_role_counts),
        }

    def get_neighbors(
        self,
        node_id: str,
        *,
        role: str | None = None,
        edge_role: str | None = None,
    ) -> list[GraphNode]:
        neighbors: list[GraphNode] = []
        for edge_id in self.adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            if edge_role and edge.role != edge_role:
                continue
            other_id = edge.target if edge.source == node_id else edge.source
            other = self.nodes.get(other_id)
            if other is None:
                continue
            if role and other.role != role:
                continue
            neighbors.append(other)
        return neighbors

    def get_neighbor_ids(
        self,
        node_id: str,
        *,
        role: str | None = None,
        edge_role: str | None = None,
    ) -> list[str]:
        return [node.node_id for node in self.get_neighbors(node_id, role=role, edge_role=edge_role)]

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        return self.get_neighbor_ids(entity_id, role="hyperedge", edge_role="link")

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        return self.get_neighbor_ids(hyperedge_id, role="entity", edge_role="link")

    def synonym_entity_ids(self, synonym_id: str) -> list[str]:
        return self.get_neighbor_ids(synonym_id, role="entity", edge_role="synonym")

    def node_chunk_ids(self, node_id: str) -> list[str]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return list(node.source_ids)

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        return self.node_chunk_ids(hyperedge_id)

    def expand_from_entities(self, entity_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        expanded: list[str] = []
        for entity_id in entity_ids:
            for hyperedge_id in self.entity_hyperedge_ids(entity_id):
                if hyperedge_id in seen:
                    continue
                seen.add(hyperedge_id)
                expanded.append(hyperedge_id)
        return expanded

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, Any]:
        hyperedge = self.nodes.get(hyperedge_id)
        if hyperedge is None:
            return {
                "hyperedge_id": hyperedge_id,
                "hyperedge_text": hyperedge_id,
                "entity_ids": [],
                "chunk_ids": [],
            }
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": hyperedge.node_id,
            "entity_ids": self.hyperedge_entity_ids(hyperedge_id),
            "chunk_ids": self.hyperedge_chunk_ids(hyperedge_id),
            "source_ids": list(hyperedge.source_ids),
            "weight": hyperedge.weight,
        }


def _collect_graphml_data(element: ET.Element, key_names: dict[str, str], namespace: dict[str, str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for data_elem in element.findall("g:data", namespace):
        key_id = data_elem.attrib["key"]
        attr_name = key_names.get(key_id, key_id)
        value = data_elem.text or ""
        if attr_name == "weight" and "weight" in payload:
            payload["weight_float"] = value
        else:
            payload[attr_name] = value
    return payload
