"""
Graph-Only Retrieval and Multi-Hop Traversal Package.
Exports query interpreter, entity resolver, traversal engine, context builder,
and high-level GraphRetriever orchestrator.
"""

from app.graph.retrieval.specs import (
    QueryIntent,
    RetrievalSpec,
    TraversalDirection,
)
from app.graph.retrieval.access_control import (
    AccessControlContext,
    AccessController,
    LocalDevAccessController,
    ScopedAccessController,
    default_access_controller,
)
from app.graph.retrieval.query_interpreter import QueryInterpreter
from app.graph.retrieval.entity_resolver import (
    EntityResolver,
    ResolutionStatus,
    EntityResolutionResult,
    EntityCandidate,
)
from app.graph.retrieval.subgraph import (
    RetrievalResult,
    RetrievedNode,
    RetrievedEdge,
    PathResult,
)
from app.graph.retrieval.traversal import GraphTraversalEngine
from app.graph.retrieval.context_builder import GraphContextBuilder
from app.graph.retrieval.retriever import GraphRetriever

__all__ = [
    "QueryIntent",
    "RetrievalSpec",
    "TraversalDirection",
    "AccessControlContext",
    "AccessController",
    "LocalDevAccessController",
    "ScopedAccessController",
    "default_access_controller",
    "QueryInterpreter",
    "EntityResolver",
    "ResolutionStatus",
    "EntityResolutionResult",
    "EntityCandidate",
    "RetrievalResult",
    "RetrievedNode",
    "RetrievedEdge",
    "PathResult",
    "GraphTraversalEngine",
    "GraphContextBuilder",
    "GraphRetriever",
]
