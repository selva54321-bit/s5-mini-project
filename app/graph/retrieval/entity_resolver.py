"""
Ambiguity-Aware Entity Resolution Module for Graph Retrieval.
Resolves query entity mentions against graph nodes using stable URNs,
qualified names, routes, file paths, and scopes. Explicitly detects and records
ambiguity rather than guessing arbitrary targets.
"""

from enum import Enum
import logging
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field

from app.graph.store import GraphStore, graph_store
from app.graph.schema import validate_node_label, ALLOWED_NODE_LABELS
from app.graph.retrieval.access_control import (
    AccessControlContext,
    AccessController,
    default_access_controller,
)

logger = logging.getLogger("graph.retrieval.resolver")


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


class EntityCandidate(BaseModel):
    urn: str
    label: str
    name: str
    qualified_name: Optional[str] = None
    file_path: Optional[str] = None
    project_urn: Optional[str] = None
    repo_urn: Optional[str] = None
    match_reason: str = "exact"


class EntityResolutionResult(BaseModel):
    mention: str
    status: ResolutionStatus
    resolved_node: Optional[Dict[str, Any]] = None
    candidates: List[EntityCandidate] = Field(default_factory=list)
    message: str = ""


class EntityResolver:
    """
    Resolves natural language or code entity mentions to verified graph entities.
    """

    def __init__(
        self,
        store: Optional[GraphStore] = None,
        access_controller: Optional[AccessController] = None,
    ):
        self.store = store or graph_store
        self.access_controller = access_controller or default_access_controller

    def resolve_mention(
        self,
        mention: str,
        entity_types: Optional[List[str]] = None,
        scope_project_urn: Optional[str] = None,
        scope_repo_urn: Optional[str] = None,
        context: Optional[AccessControlContext] = None,
    ) -> EntityResolutionResult:
        """
        Resolves a single mention to a graph entity node.
        """
        ctx = context or AccessControlContext()
        m = mention.strip()
        if not m:
            return EntityResolutionResult(
                mention=mention,
                status=ResolutionStatus.NOT_FOUND,
                message="Empty entity mention provided.",
            )

        # 1. Exact URN match
        if m.startswith("urn:"):
            node = self.store.get_node(m)
            if node and self.access_controller.can_access_node(node, ctx):
                labels = node.get("_labels", [])
                label = labels[0] if labels else "Unknown"
                return EntityResolutionResult(
                    mention=m,
                    status=ResolutionStatus.RESOLVED,
                    resolved_node=node,
                    candidates=[
                        EntityCandidate(
                            urn=node["id"],
                            label=label,
                            name=node.get("name", m),
                            qualified_name=node.get("qualified_name"),
                            file_path=node.get("file_path") or node.get("rel_path"),
                            match_reason="exact_urn",
                        )
                    ],
                    message=f"Exact match on URN {m}",
                )
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.NOT_FOUND,
                message=f"No entity found with URN {m} or access denied.",
            )

        # 2. Database search for candidate matches
        candidates = self._find_candidates(
            mention=m,
            entity_types=entity_types,
            scope_project_urn=scope_project_urn,
            scope_repo_urn=scope_repo_urn,
            context=ctx,
        )

        if not candidates:
            # Try fuzzy/contains search for useful candidate suggestions
            suggestions = self._find_partial_suggestions(m, entity_types, ctx)
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.NOT_FOUND,
                candidates=suggestions,
                message=f"Could not resolve entity mention '{m}'. "
                + (f"Did you mean one of: {[s.name for s in suggestions]}?" if suggestions else "No close candidates found."),
            )

        # Exact match filtering
        exact_matches = [c for c in candidates if c.match_reason in ("exact_qualified_name", "exact_name", "exact_route", "exact_file_path")]

        # If exactly one exact match exists
        if len(exact_matches) == 1:
            best = exact_matches[0]
            node = self.store.get_node(best.urn)
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.RESOLVED,
                resolved_node=node,
                candidates=[best],
                message=f"Resolved '{m}' to {best.urn} via {best.match_reason}",
            )

        # If multiple exact matches exist across different files/repos, detect AMBIGUITY
        if len(exact_matches) > 1:
            # Check if one matches qualified name exclusively
            qn_matches = [c for c in exact_matches if c.match_reason == "exact_qualified_name"]
            if len(qn_matches) == 1:
                best = qn_matches[0]
                node = self.store.get_node(best.urn)
                return EntityResolutionResult(
                    mention=m,
                    status=ResolutionStatus.RESOLVED,
                    resolved_node=node,
                    candidates=[best],
                    message=f"Resolved '{m}' uniquely to {best.urn} by qualified name.",
                )

            # Truly ambiguous
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.AMBIGUOUS,
                candidates=exact_matches,
                message=f"Ambiguous entity mention '{m}'. Multiple matching entities found ({len(exact_matches)}). "
                f"Candidate URNs: {[c.urn for c in exact_matches]}. Please disambiguate with full qualified name or path.",
            )

        # Case-insensitive matches
        ci_matches = [c for c in candidates if c.match_reason == "case_insensitive_name"]
        if len(ci_matches) == 1:
            best = ci_matches[0]
            node = self.store.get_node(best.urn)
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.RESOLVED,
                resolved_node=node,
                candidates=[best],
                message=f"Resolved '{m}' to {best.urn} via case-insensitive name match.",
            )
        elif len(ci_matches) > 1:
            return EntityResolutionResult(
                mention=m,
                status=ResolutionStatus.AMBIGUOUS,
                candidates=ci_matches,
                message=f"Ambiguous entity mention '{m}'. Multiple case-insensitive matches found: {[c.urn for c in ci_matches]}.",
            )

        return EntityResolutionResult(
            mention=m,
            status=ResolutionStatus.NOT_FOUND,
            candidates=candidates[:5],
            message=f"Entity mention '{m}' could not be definitively resolved.",
        )

    def _find_candidates(
        self,
        mention: str,
        entity_types: Optional[List[str]],
        scope_project_urn: Optional[str],
        scope_repo_urn: Optional[str],
        context: AccessControlContext,
    ) -> List[EntityCandidate]:
        """
        Queries Neo4j for nodes matching the mention on name, qualified_name, route, or file_path.
        """
        type_clauses = ""
        if entity_types:
            valid_types = [validate_node_label(t) for t in entity_types if t in ALLOWED_NODE_LABELS]
            if valid_types:
                # MATCH (n) WHERE any(l IN labels(n) WHERE l IN $valid_types)
                type_clauses = "AND any(l IN labels(n) WHERE l IN $valid_types)"

        cypher = f"""
        MATCH (n)
        WHERE (
            n.id = $mention OR
            n.qualified_name = $mention OR
            n.name = $mention OR
            n.route = $mention OR
            n.file_path = $mention OR
            n.rel_path = $mention OR
            toLower(n.name) = toLower($mention)
        )
        {type_clauses}
        RETURN n, labels(n) AS labels
        LIMIT 25
        """

        params: Dict[str, Any] = {"mention": mention}
        if entity_types:
            params["valid_types"] = [validate_node_label(t) for t in entity_types if t in ALLOWED_NODE_LABELS]

        def _work(tx):
            res = tx.run(cypher, **params)
            candidates = []
            for record in res:
                node = dict(record["n"])
                labels = record["labels"]
                if not self.access_controller.can_access_node(node, context):
                    continue

                # Scope check
                if scope_project_urn:
                    node_proj = node.get("project_urn") or node.get("provenance_source_project_id")
                    if node_proj and node_proj != scope_project_urn:
                        continue
                if scope_repo_urn:
                    node_repo = node.get("repo_urn") or node.get("provenance_source_repo_id")
                    if node_repo and node_repo != scope_repo_urn:
                        continue

                label = labels[0] if labels else "Unknown"
                reason = "exact_name"
                if node.get("qualified_name") == mention:
                    reason = "exact_qualified_name"
                elif node.get("route") == mention:
                    reason = "exact_route"
                elif node.get("file_path") == mention or node.get("rel_path") == mention:
                    reason = "exact_file_path"
                elif (node.get("name") or "").lower() == mention.lower():
                    reason = "case_insensitive_name"

                candidates.append(
                    EntityCandidate(
                        urn=node["id"],
                        label=label,
                        name=node.get("name", ""),
                        qualified_name=node.get("qualified_name"),
                        file_path=node.get("file_path") or node.get("rel_path"),
                        project_urn=node.get("project_urn"),
                        repo_urn=node.get("repo_urn"),
                        match_reason=reason,
                    )
                )
            return candidates

        return self.store.db.execute_read(_work)

    def _find_partial_suggestions(
        self,
        mention: str,
        entity_types: Optional[List[str]],
        context: AccessControlContext,
    ) -> List[EntityCandidate]:
        """
        Finds up to 5 prefix or substring suggestions when exact match fails.
        """
        cypher = """
        MATCH (n)
        WHERE (toLower(n.name) CONTAINS toLower($mention) OR toLower(n.id) CONTAINS toLower($mention))
        RETURN n, labels(n) AS labels
        LIMIT 5
        """

        def _work(tx):
            res = tx.run(cypher, mention=mention)
            suggestions = []
            for record in res:
                node = dict(record["n"])
                labels = record["labels"]
                if not self.access_controller.can_access_node(node, context):
                    continue
                label = labels[0] if labels else "Unknown"
                suggestions.append(
                    EntityCandidate(
                        urn=node["id"],
                        label=label,
                        name=node.get("name", ""),
                        qualified_name=node.get("qualified_name"),
                        file_path=node.get("file_path") or node.get("rel_path"),
                        match_reason="substring_candidate",
                    )
                )
            return suggestions

        return self.store.db.execute_read(_work)
