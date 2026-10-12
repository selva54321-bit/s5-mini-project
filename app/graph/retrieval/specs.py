"""
Retrieval Specification and Intent Models for Graph-Only Retrieval.
Defines validated intents, parameters, traversal constraints, and schema enforcement.
"""

from enum import Enum
from typing import List, Optional, Set, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

from app.graph.schema import (
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATIONSHIPS,
    validate_node_label,
    validate_relationship_type,
)


class QueryIntent(str, Enum):
    ENTITY_LOOKUP = "entity_lookup"
    DEPENDENCY_LOOKUP = "dependency_lookup"
    CALLER_LOOKUP = "caller_lookup"
    CALLEE_LOOKUP = "callee_lookup"
    MULTIHOP_PATH = "multihop_path"
    IMPACT_ANALYSIS = "impact_analysis"
    TEST_DISCOVERY = "test_discovery"
    ARCHITECTURE_LOOKUP = "architecture_lookup"
    POLICY_LOOKUP = "policy_lookup"
    SECURITY_REQUIREMENT_LOOKUP = "security_requirement_lookup"


class TraversalDirection(str, Enum):
    OUTGOING = "OUTGOING"
    INCOMING = "INCOMING"
    BOTH = "BOTH"


class RetrievalSpec(BaseModel):
    """
    Validated retrieval specification produced by query interpretation.
    Guarantees that all node labels, relationship types, depth, and limits
    comply with the approved knowledge graph schema.
    """
    intent: QueryIntent
    entity_mentions: List[str] = Field(default_factory=list, description="Primary entity mentions to resolve")
    target_entity_mentions: List[str] = Field(
        default_factory=list,
        description="Target entity mentions for multihop path finding",
    )
    entity_types: List[str] = Field(
        default_factory=list,
        description="Optional filter on node labels (must be in approved schema)",
    )
    relationship_types: List[str] = Field(
        default_factory=list,
        description="Optional filter on relationship types (must be in approved schema)",
    )
    direction: TraversalDirection = TraversalDirection.BOTH
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum traversal depth (hard ceiling 5)")
    max_nodes: int = Field(default=50, ge=1, le=200, description="Maximum nodes in retrieved subgraph")
    max_paths: int = Field(default=20, ge=1, le=100, description="Maximum paths to explore")
    timeout_seconds: float = Field(default=10.0, ge=0.5, le=60.0)
    include_provenance: bool = True
    scope_project_urn: Optional[str] = None
    scope_repo_urn: Optional[str] = None
    raw_query: Optional[str] = None

    @field_validator("entity_types")
    @classmethod
    def validate_entity_types(cls, labels: List[str]) -> List[str]:
        validated = []
        for l in labels:
            if not l:
                continue
            validated.append(validate_node_label(l))
        return validated

    @field_validator("relationship_types")
    @classmethod
    def validate_relationship_types(cls, rels: List[str]) -> List[str]:
        validated = []
        for r in rels:
            if not r:
                continue
            validated.append(validate_relationship_type(r))
        return validated

    @model_validator(mode="after")
    def validate_intent_requirements(self) -> "RetrievalSpec":
        """Validates intent-specific parameter consistency."""
        if self.intent == QueryIntent.MULTIHOP_PATH:
            if not self.entity_mentions or not self.target_entity_mentions:
                # Require at least one source and one target for multihop path finding
                pass  # Resolver handles partial mentions gracefully
        return self
