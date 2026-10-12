"""
Structured Subgraph Result Models for Graph Retrieval.
Defines deduplicated representations of retrieved nodes, relationships,
ordered multi-hop paths, hop distances, and provenance metadata.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.graph.retrieval.specs import QueryIntent


class RetrievedNode(BaseModel):
    id: str
    label: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    hop_distance: int = 0
    provenance_type: Optional[str] = None
    provenance_file: Optional[str] = None
    provenance_lines: Optional[str] = None
    confidence: float = 1.0


class RetrievedEdge(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    direction: str = "OUTGOING"
    hop: int = 1
    provenance_type: Optional[str] = None
    confidence: float = 1.0


class PathResult(BaseModel):
    path_id: str
    hop_length: int
    nodes: List[RetrievedNode]
    relationships: List[RetrievedEdge]
    summary: str


class RetrievalResult(BaseModel):
    """
    Structured outcome of deterministic multi-hop graph retrieval.
    Guarantees that nodes and relationships are deduplicated,
    hop distances and directions are preserved, and ambiguities are documented.
    """
    intent: QueryIntent
    starting_entities: List[RetrievedNode] = Field(default_factory=list)
    target_entities: List[RetrievedNode] = Field(default_factory=list)
    nodes: Dict[str, RetrievedNode] = Field(default_factory=dict, description="Nodes indexed by URN")
    relationships: List[RetrievedEdge] = Field(default_factory=list)
    paths: List[PathResult] = Field(default_factory=list)
    ambiguities: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_references: List[str] = Field(default_factory=list)
    is_truncated: bool = False
    warnings: List[str] = Field(default_factory=list)
    provenance_summary: Dict[str, int] = Field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_relationships(self) -> int:
        return len(self.relationships)

    def add_node(self, node: RetrievedNode) -> None:
        if node.id in self.nodes:
            # Update to shorter hop distance if found via shorter path
            if node.hop_distance < self.nodes[node.id].hop_distance:
                self.nodes[node.id].hop_distance = node.hop_distance
        else:
            self.nodes[node.id] = node

        # Update provenance stats
        p_type = node.provenance_type or "UNKNOWN"
        self.provenance_summary[p_type] = self.provenance_summary.get(p_type, 0) + 1

    def add_relationship(self, edge: RetrievedEdge) -> None:
        # Deduplicate edges by (source_id, rel_type, target_id)
        exists = any(
            e.source_id == edge.source_id and e.target_id == edge.target_id and e.rel_type == edge.rel_type
            for e in self.relationships
        )
        if not exists:
            self.relationships.append(edge)
            p_type = edge.provenance_type or "UNKNOWN"
            self.provenance_summary[p_type] = self.provenance_summary.get(p_type, 0) + 1
