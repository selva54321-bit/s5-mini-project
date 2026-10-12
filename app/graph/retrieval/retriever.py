"""
Intent-Specific Graph Retrieval Orchestrator.
Coordinates query interpretation, ambiguity-aware entity resolution,
deterministic Cypher traversal, and structured Subgraph extraction.
Strictly graph-only: zero embeddings, zero vector search.
"""

import logging
from typing import Union, Optional, List, Dict, Any

from app.graph.store import GraphStore, graph_store
from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.retrieval.query_interpreter import QueryInterpreter
from app.graph.retrieval.entity_resolver import (
    EntityResolver,
    ResolutionStatus,
    EntityResolutionResult,
)
from app.graph.retrieval.traversal import (
    GraphTraversalEngine,
    _extract_node_from_record,
)
from app.graph.retrieval.subgraph import (
    RetrievalResult,
    RetrievedNode,
    RetrievedEdge,
    PathResult,
)
from app.graph.retrieval.access_control import (
    AccessControlContext,
    AccessController,
    default_access_controller,
)

logger = logging.getLogger("graph.retrieval.orchestrator")


class GraphRetriever:
    """
    Core graph-only knowledge intelligence retrieval engine.
    Retrieves facts, dependencies, callers, impact blast radius, tests, and policies
    by traversing explicit relationships in Neo4j.
    """

    def __init__(
        self,
        store: Optional[GraphStore] = None,
        interpreter: Optional[QueryInterpreter] = None,
        resolver: Optional[EntityResolver] = None,
        traversal_engine: Optional[GraphTraversalEngine] = None,
        access_controller: Optional[AccessController] = None,
    ):
        self.store = store or graph_store
        self.access_controller = access_controller or default_access_controller
        self.interpreter = interpreter or QueryInterpreter()
        self.resolver = resolver or EntityResolver(self.store, self.access_controller)
        self.traversal_engine = traversal_engine or GraphTraversalEngine(self.store, self.access_controller)

    def retrieve(
        self,
        query_or_spec: Union[str, RetrievalSpec],
        context: Optional[AccessControlContext] = None,
    ) -> RetrievalResult:
        """
        Executes end-to-end graph retrieval for a query or specification.
        """
        ctx = context or AccessControlContext()
        self.access_controller.assert_service_exposure_permitted(ctx)

        # 1. Obtain validated RetrievalSpec
        if isinstance(query_or_spec, str):
            spec = self.interpreter.interpret(query_or_spec)
        else:
            spec = query_or_spec

        result = RetrievalResult(intent=spec.intent)

        # 2. Resolve primary entity mentions
        resolved_starts: List[RetrievedNode] = []
        for mention in spec.entity_mentions:
            res: EntityResolutionResult = self.resolver.resolve_mention(
                mention=mention,
                entity_types=spec.entity_types,
                scope_project_urn=spec.scope_project_urn,
                scope_repo_urn=spec.scope_repo_urn,
                context=ctx,
            )

            if res.status == ResolutionStatus.RESOLVED and res.resolved_node:
                r_node = _extract_node_from_record(res.resolved_node, res.resolved_node.get("_labels", []), hop=0)
                resolved_starts.append(r_node)
                result.add_node(r_node)
                result.starting_entities.append(r_node)
            elif res.status == ResolutionStatus.AMBIGUOUS:
                result.ambiguities.append({
                    "mention": mention,
                    "candidates": [c.model_dump() for c in res.candidates],
                    "message": res.message,
                })
                result.warnings.append(f"Ambiguous mention '{mention}': {res.message}")
            else:
                result.unresolved_references.append(mention)
                if res.candidates:
                    result.warnings.append(
                        f"Unresolved mention '{mention}'. Potential candidates: {[c.name for c in res.candidates]}"
                    )
                else:
                    result.warnings.append(f"Entity mention '{mention}' could not be found in the knowledge graph.")

        # If no starting entities could be resolved and mentions were provided
        if not resolved_starts and spec.entity_mentions:
            result.warnings.append("No starting entities could be resolved. Graph traversal cannot proceed.")
            return result

        # 3. Intent-Specific Retrieval Dispatch
        if spec.intent == QueryIntent.ENTITY_LOOKUP:
            self._retrieve_entity_lookup(resolved_starts, spec, result, ctx)

        elif spec.intent == QueryIntent.DEPENDENCY_LOOKUP:
            self._retrieve_dependencies(resolved_starts, spec, result, ctx)

        elif spec.intent in (QueryIntent.CALLER_LOOKUP, QueryIntent.CALLEE_LOOKUP):
            self._retrieve_callers_and_callees(resolved_starts, spec, result, ctx)

        elif spec.intent == QueryIntent.MULTIHOP_PATH:
            self._retrieve_multihop_path(resolved_starts, spec, result, ctx)

        elif spec.intent == QueryIntent.IMPACT_ANALYSIS:
            self._retrieve_impact_analysis(resolved_starts, spec, result, ctx)

        elif spec.intent == QueryIntent.TEST_DISCOVERY:
            self._retrieve_tests(resolved_starts, spec, result, ctx)

        elif spec.intent in (QueryIntent.ARCHITECTURE_LOOKUP, QueryIntent.POLICY_LOOKUP, QueryIntent.SECURITY_REQUIREMENT_LOOKUP):
            self._retrieve_architecture_and_policies(resolved_starts, spec, result, ctx)

        # 4. Enforce result truncation constraints
        if len(result.nodes) > spec.max_nodes:
            result.is_truncated = True
            # Keep starting entities and sort by hop distance, truncate excess
            sorted_nodes = sorted(result.nodes.values(), key=lambda n: n.hop_distance)
            retained_nodes = {n.id: n for n in sorted_nodes[:spec.max_nodes]}
            result.nodes = retained_nodes
            # Retain only edges whose endpoints are in retained_nodes
            result.relationships = [
                e for e in result.relationships if e.source_id in retained_nodes and e.target_id in retained_nodes
            ]
            result.warnings.append(
                f"Result subgraph truncated to max_nodes ceiling ({spec.max_nodes} of {len(sorted_nodes)} nodes retained)."
            )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Intent-Specific Retrieval Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_entity_lookup(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Retrieves direct 1-hop environment for starting entities."""
        for start in starts:
            neighbors = self.traversal_engine.get_direct_neighbors(
                source_id=start.id,
                direction=spec.direction,
                rel_types=spec.relationship_types,
                limit=spec.max_nodes,
                context=ctx,
            )
            for item in neighbors:
                result.add_node(item["target"])
                edge = RetrievedEdge(
                    source_id=item["rel_source_id"],
                    target_id=item["rel_target_id"],
                    rel_type=item["rel_type"],
                    properties=item["rel_props"],
                    direction="OUTGOING" if item["rel_source_id"] == start.id else "INCOMING",
                    hop=1,
                    provenance_type=item["rel_props"].get("provenance_type"),
                    confidence=float(item["rel_props"].get("confidence") or 1.0),
                )
                result.add_relationship(edge)

    def _retrieve_dependencies(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Traverses DEPENDS_ON, IMPORTS, CONTAINS edges up to max_depth."""
        rel_types = spec.relationship_types or ["DEPENDS_ON", "IMPORTS", "CONTAINS"]
        for start in starts:
            paths = self.traversal_engine.find_bounded_paths(
                source_id=start.id,
                direction=spec.direction,
                rel_types=rel_types,
                max_depth=spec.max_depth,
                max_paths=spec.max_paths,
                context=ctx,
            )
            for p in paths:
                result.paths.append(p)
                for node in p.nodes:
                    result.add_node(node)
                for edge in p.relationships:
                    result.add_relationship(edge)

    def _retrieve_callers_and_callees(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Retrieves statically established CALLS relationships."""
        rel_types = ["CALLS"]
        for start in starts:
            paths = self.traversal_engine.find_bounded_paths(
                source_id=start.id,
                direction=spec.direction,
                rel_types=rel_types,
                max_depth=spec.max_depth,
                max_paths=spec.max_paths,
                context=ctx,
            )
            for p in paths:
                result.paths.append(p)
                for node in p.nodes:
                    result.add_node(node)
                for edge in p.relationships:
                    result.add_relationship(edge)

    def _retrieve_multihop_path(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Discovers paths connecting source and target entities."""
        targets: List[RetrievedNode] = []
        for tgt_mention in spec.target_entity_mentions:
            tgt_res = self.resolver.resolve_mention(
                mention=tgt_mention,
                scope_project_urn=spec.scope_project_urn,
                scope_repo_urn=spec.scope_repo_urn,
                context=ctx,
            )
            if tgt_res.status == ResolutionStatus.RESOLVED and tgt_res.resolved_node:
                tgt_node = _extract_node_from_record(tgt_res.resolved_node, tgt_res.resolved_node.get("_labels", []), hop=0)
                targets.append(tgt_node)
                result.add_node(tgt_node)
                result.target_entities.append(tgt_node)
            elif tgt_res.status == ResolutionStatus.AMBIGUOUS:
                result.ambiguities.append({
                    "mention": tgt_mention,
                    "candidates": [c.model_dump() for c in tgt_res.candidates],
                    "message": tgt_res.message,
                })
            else:
                result.unresolved_references.append(tgt_mention)

        if not targets:
            result.warnings.append("Target entities could not be resolved for multihop path finding.")
            return

        for start in starts:
            for target in targets:
                paths = self.traversal_engine.find_shortest_paths_between(
                    source_id=start.id,
                    target_id=target.id,
                    rel_types=spec.relationship_types,
                    max_depth=spec.max_depth,
                    max_paths=spec.max_paths,
                    context=ctx,
                )
                if not paths:
                    result.warnings.append(
                        f"No path found connecting '{start.name}' to '{target.name}' within {spec.max_depth} hops."
                    )
                for p in paths:
                    result.paths.append(p)
                    for node in p.nodes:
                        result.add_node(node)
                    for edge in p.relationships:
                        result.add_relationship(edge)

    def _retrieve_impact_analysis(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Finds potentially affected components along reverse dependency/call chains."""
        result.warnings.append(
            "Impact Notice: Static graph reachability denotes potential dependency exposure; "
            "it does not guarantee runtime execution along every path."
        )
        for start in starts:
            paths = self.traversal_engine.find_impact_chain(
                source_id=start.id,
                max_depth=spec.max_depth,
                max_paths=spec.max_paths,
                context=ctx,
            )
            for p in paths:
                result.paths.append(p)
                for node in p.nodes:
                    result.add_node(node)
                for edge in p.relationships:
                    result.add_relationship(edge)

    def _retrieve_tests(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Retrieves tests connected via TESTED_BY or CALLS edges."""
        for start in starts:
            rows = self.traversal_engine.find_tests_for(
                source_id=start.id,
                limit=spec.max_nodes,
                context=ctx,
            )
            if not rows:
                result.warnings.append(f"No automated tests found directly connected to '{start.name}'.")
            for item in rows:
                test_node = item["test"]
                result.add_node(test_node)
                edge = RetrievedEdge(
                    source_id=test_node.id,
                    target_id=start.id,
                    rel_type=item["rel_type"],
                    properties=item["rel_props"],
                    direction="INCOMING",
                    hop=1,
                    provenance_type=item["rel_props"].get("provenance_type"),
                    confidence=float(item["rel_props"].get("confidence") or 1.0),
                )
                result.add_relationship(edge)

    def _retrieve_architecture_and_policies(
        self,
        starts: List[RetrievedNode],
        spec: RetrievalSpec,
        result: RetrievalResult,
        ctx: AccessControlContext,
    ) -> None:
        """Retrieves connected architecture components, documents, policies, and security rules."""
        for start in starts:
            rows = self.traversal_engine.find_policies_and_governance(
                source_id=start.id,
                limit=spec.max_nodes,
                context=ctx,
            )
            for item in rows:
                t_node = item["target"]
                result.add_node(t_node)
                edge = RetrievedEdge(
                    source_id=item["rel_source_id"],
                    target_id=item["rel_target_id"],
                    rel_type=item["rel_type"],
                    properties=item["rel_props"],
                    direction="OUTGOING" if item["rel_source_id"] == start.id else "INCOMING",
                    hop=1,
                    provenance_type=item["rel_props"].get("provenance_type"),
                    confidence=float(item["rel_props"].get("confidence") or 1.0),
                )
                result.add_relationship(edge)
