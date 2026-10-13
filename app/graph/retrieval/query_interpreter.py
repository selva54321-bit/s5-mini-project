"""
Query Interpretation Module.
Converts natural language user questions into validated RetrievalSpec objects.
Uses deterministic regex and heuristic rules first, with optional LLM interpretation
strictly validated against approved schema intents, entity labels, and relationship types.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.schema import (
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATIONSHIPS,
    validate_node_label,
    validate_relationship_type,
)

logger = logging.getLogger("graph.retrieval.interpreter")


class QueryInterpreter:
    """
    Interprets natural language queries into deterministic, validated retrieval specifications.
    Guarantees no graph facts are created or altered.
    """

    def __init__(self, use_llm_fallback: bool = False):
        self.use_llm_fallback = use_llm_fallback

    def interpret(
        self,
        query: str,
        scope_project_urn: Optional[str] = None,
        scope_repo_urn: Optional[str] = None,
    ) -> RetrievalSpec:
        """
        Main entry point for converting a question into a validated RetrievalSpec.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        q_clean = query.strip()
        spec = self._interpret_deterministic(q_clean)

        if spec is None and self.use_llm_fallback:
            spec = self._interpret_llm(q_clean)

        if spec is None:
            # Fallback to general entity lookup
            mentions = self._extract_entity_mentions(q_clean)
            spec = RetrievalSpec(
                intent=QueryIntent.ENTITY_LOOKUP,
                entity_mentions=mentions,
                raw_query=q_clean,
            )

        # Attach scopes and query text
        spec.scope_project_urn = scope_project_urn
        spec.scope_repo_urn = scope_repo_urn
        spec.raw_query = q_clean
        return spec

    def _interpret_deterministic(self, query: str) -> Optional[RetrievalSpec]:
        """
        Deterministic pattern matching for intents and parameters.
        """
        q_lower = query.lower()

        # 1. Multi-hop Path Discovery between two entities
        # e.g., "Find path between Service A and Database B", "How is A connected to B?"
        path_match = re.search(
            r"(?:path|connection|link)\s+between\s+[`'\"]?([^`'\"]+?)[`'\"]?\s+and\s+[`'\"]?([^`'\"]+?)[`'\"]?(?:\?|$|\.|\s)",
            query,
            re.IGNORECASE,
        ) or re.search(
            r"how\s+is\s+[`'\"]?([^`'\"]+?)[`'\"]?\s+connected\s+to\s+[`'\"]?([^`'\"]+?)[`'\"]?(?:\?|$|\.|\s)",
            query,
            re.IGNORECASE,
        )
        if path_match:
            src, tgt = path_match.group(1).strip(), path_match.group(2).strip()
            return RetrievalSpec(
                intent=QueryIntent.MULTIHOP_PATH,
                entity_mentions=[src],
                target_entity_mentions=[tgt],
                direction=TraversalDirection.BOTH,
                max_depth=3,
                max_paths=15,
            )

        # 2. Impact Analysis (blast radius / ripple effects)
        # e.g., "What will be affected if I change function login?", "Impact of modifying AuthService"
        if any(kw in q_lower for kw in ["impact", "blast radius", "affected if", "breaks if", "what breaks", "changing ", "modifying "]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.IMPACT_ANALYSIS,
                entity_mentions=mentions,
                direction=TraversalDirection.INCOMING,  # Who depends on or calls the target
                relationship_types=["CALLS", "DEPENDS_ON", "IMPORTS"],
                max_depth=3,
                max_nodes=60,
            )

        # 3. Caller Lookup (who calls X)
        if any(kw in q_lower for kw in ["who calls", "callers of", "what calls", "where is it called", "is called by"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.CALLER_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.INCOMING,
                relationship_types=["CALLS"],
                entity_types=["Function", "Class"],
                max_depth=2,
            )

        # 4. Callee Lookup (what does X call)
        if any(kw in q_lower for kw in ["what does it call", "what does this call", "callees of", "functions called by", "invoked by"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.CALLEE_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.OUTGOING,
                relationship_types=["CALLS"],
                entity_types=["Function", "Class"],
                max_depth=2,
            )

        # 5. Dependency Lookup
        if any(kw in q_lower for kw in ["depend on", "dependencies", "what packages", "what does it import", "imports of"]):
            mentions = self._extract_entity_mentions(query)
            # Check if asking what depends on X (incoming) or what X depends on (outgoing)
            direction = TraversalDirection.INCOMING if "what depends on" in q_lower or "depend on it" in q_lower else TraversalDirection.OUTGOING
            return RetrievalSpec(
                intent=QueryIntent.DEPENDENCY_LOOKUP,
                entity_mentions=mentions,
                direction=direction,
                relationship_types=["DEPENDS_ON", "IMPORTS", "CONTAINS"],
                max_depth=2,
            )

        # 6. Test Discovery
        if any(kw in q_lower for kw in ["test", "tests for", "tests covering", "how is it tested", "test suite"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.TEST_DISCOVERY,
                entity_mentions=mentions,
                direction=TraversalDirection.BOTH,
                relationship_types=["TESTED_BY", "CALLS"],
                max_depth=2,
            )

        # 7. Security Requirement Lookup
        if any(kw in q_lower for kw in ["security rule", "security requirement", "security policy", "gatekeeper", "data retention", "cve"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.SECURITY_REQUIREMENT_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.BOTH,
                relationship_types=["REQUIRES", "GOVERNED_BY", "APPLIES_TO"],
                max_depth=2,
            )

        # 8. Policy & Coding Standard Lookup
        if any(kw in q_lower for kw in ["policy", "policies", "coding standard", "compliance", "governed by", "standard"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.POLICY_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.BOTH,
                relationship_types=["GOVERNED_BY", "APPLIES_TO", "REQUIRES"],
                max_depth=2,
            )

        # 9. Architecture Component Lookup
        # API/endpoints scoped with "in <repository>" are repository questions,
        # not architecture questions.  This also tolerates natural phrasing such
        # as "in eqrev what weather API is used".
        if any(term in q_lower for term in ("api", "endpoint")):
            scoped_repo = re.search(
                r"\bin\s+(?:the\s+)?(?:repo(?:sitory)?\s+)?[`'\"]*([A-Za-z0-9_.-]+)[`'\"]*",
                query,
                re.IGNORECASE,
            )
            if scoped_repo:
                repo_name = scoped_repo.group(1).strip(" ?.,;")
                if repo_name.lower() not in {"the", "project", "repository", "repo", "codebase"}:
                    return RetrievalSpec(
                        intent=QueryIntent.REPOSITORY_LOOKUP,
                        entity_mentions=[repo_name],
                        direction=TraversalDirection.OUTGOING,
                        relationship_types=["CONTAINS", "EXPOSES"],
                        entity_types=["Repository"],
                        max_depth=2,
                        max_nodes=100,
                    )

        if any(kw in q_lower for kw in ["architecture", "component", "service", "tier", "database", "api route", "exposes"]):
            mentions = self._extract_entity_mentions(query)
            return RetrievalSpec(
                intent=QueryIntent.ARCHITECTURE_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.BOTH,
                relationship_types=["DOCUMENTED_BY", "EXPOSES", "CONTAINS", "BELONGS_TO"],
                max_depth=2,
            )

        # 10. Repository Lookup (files in X repo, structure of X, what's in repository X)
        repo_keywords = ["repository", "repo", "codebase", "project"]
        repo_content_keywords = [
            "files in", "files are in", "structure of", "what's in", "contents of",
            "modules in", "classes in", "functions in", "endpoints in", "apis in",
            "functions exist in", "classes exist in", "endpoints exist in",
            "what functions", "what classes", "what apis", "what endpoints",
            "what api", "api endpoints", "show me", "tell me about repo", "list all",
        ]
        if any(kw in q_lower for kw in repo_content_keywords) or (
            any(kw in q_lower for kw in repo_keywords) and any(kw in q_lower for kw in ["what", "show", "list", "describe", "tell"])
        ):
            # Generic scope words are not graph entities.  Leave the scope empty
            # so the retriever can identify the accessible repositories instead.
            generic_scope_terms = {"project", "repository", "repo", "codebase", "endpoints", "apis", "api"}
            mentions = [
                mention for mention in self._extract_entity_mentions(query)
                if mention.lower() not in generic_scope_terms
            ]
            # Also try to extract repo name from common patterns
            repo_patterns = [
                r"(?:files?|functions?|classes|endpoints?|apis?)\s+(?:in|are\s+in|exist\s+in|of)\s+(?:the\s+)?(?:repo(?:sitory)?\s+)?[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*",
                r"(?:structure|contents?)\s+of\s+(?:the\s+)?(?:repo(?:sitory)?\s+)?[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*",
                r"(?:repo(?:sitory)?|codebase|project)\s+[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*",
                r"[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*\s+(?:repo(?:sitory)?|codebase|project)",
                r"(?:what's|what\s+is)\s+in\s+[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*",
                r"in\s+[`'\"]*([A-Za-z0-9_\-\.]+)[`'\"]*\s*\??$",
            ]
            for pat in repo_patterns:
                repo_match = re.search(pat, query, re.IGNORECASE)
                if repo_match:
                    repo_name = repo_match.group(1).strip(" ?.,;")
                    if repo_name and repo_name.lower() not in {"the", "a", "this", "my", "our"} and repo_name not in mentions:
                        mentions.insert(0, repo_name)
                        break
            return RetrievalSpec(
                intent=QueryIntent.REPOSITORY_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.OUTGOING,
                relationship_types=["CONTAINS"],
                entity_types=["Repository"],
                max_depth=2,
                max_nodes=100,
            )

        # 11. Entity Lookup
        mentions = self._extract_entity_mentions(query)
        if mentions:
            return RetrievalSpec(
                intent=QueryIntent.ENTITY_LOOKUP,
                entity_mentions=mentions,
                direction=TraversalDirection.BOTH,
                max_depth=1,
            )

        return None

    def _extract_entity_mentions(self, text: str) -> List[str]:
        """
        Extracts candidate entity mentions using backticks, quotes, qualified symbols,
        keywords, and Pascal/snake_case tokens.
        """
        mentions: List[str] = []

        # 1. Backtick or quoted terms: `my_fn`, 'AuthService', "users_table"
        quoted = re.findall(r"[`'\"]([^`'\"]+)[`'\"]", text)
        for q in quoted:
            val = q.strip()
            if val and len(val) > 1 and val not in mentions:
                mentions.append(val)

        # 2. URN identifiers: urn:...
        urns = re.findall(r"\b(urn:[a-zA-Z0-9_\:\-\.\/]+)\b", text)
        for u in urns:
            if u not in mentions:
                mentions.append(u)

        # 3. Dotted qualified names: auth.security.authenticate, app.routes.api
        dotted = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b", text)
        for d in dotted:
            if d not in mentions:
                mentions.append(d)

        # 4. Contextual verb/noun targets: who calls <target>, impact of <target>, tests for <target>
        target_patterns = [
            r"(?:who\s+calls|what\s+calls|callers\s+of)\s+([A-Za-z0-9_\-\.\/]+)",
            r"(?:callees\s+of|invoked\s+by)\s+([A-Za-z0-9_\-\.\/]+)",
            r"(?:impact\s+of|blast\s+radius\s+of|modifying|changing)\s+(?:function\s+|class\s+|service\s+)?([A-Za-z0-9_\-\.\/]+)",
            r"(?:tests\s+(?:for|covering)|test\s+suite\s+for)\s+([A-Za-z0-9_\-\.\/]+)",
            r"(?:policies\s+govern(?:ing)?|governed\s+by)\s+([A-Za-z0-9_\-\.\/]+)",
        ]
        for pat in target_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                val = m.strip(" ?.,;")
                if val and val not in mentions:
                    mentions.append(val)

        # 5. Explicit keywords: function foo, class Bar, service baz, API /route
        kw_matches = re.findall(
            r"\b(?:function|fn|class|service|api|component|table|file|policy|repo)\s+([A-Za-z0-9_\-\.\/]+)\b",
            text,
            re.IGNORECASE,
        )
        for m in kw_matches:
            val = m.strip(" ?.,;")
            if val and len(val) > 1 and val not in mentions:
                mentions.append(val)

        # 6. HTTP endpoints: /api/v1/...
        routes = re.findall(r"(\/(?:api\/)?[A-Za-z0-9_\-\.\/]+)", text)
        for r in routes:
            clean_r = r.strip(" ?.,;")
            if len(clean_r) > 2 and clean_r not in mentions:
                mentions.append(clean_r)

        # 7. Fallback: Identifier-like tokens (PascalCase, snake_case, or standalone names > 2 chars)
        stopwords = {
            "what", "which", "where", "when", "why", "who", "whom", "whose", "how",
            "does", "have", "this", "that", "these", "those", "show", "find", "list",
            "lookup", "describe", "between", "path", "connection", "impact", "tests",
            "calls", "depend", "dependencies", "service", "function", "class",
            "about", "tell", "please", "give", "provide", "from", "with", "into",
            "the", "are", "files", "file", "repo", "repository", "project",
            "codebase", "structure", "contents", "modules", "exist",
            "in", "is", "used", "here", "api", "apis", "endpoint", "endpoints",
        }
        candidate_tokens = re.findall(r"\b([A-Z][a-zA-Z0-9]+|[a-z0-9]+_[a-z0-9_]+)\b", text)
        for t in candidate_tokens:
            if t.lower() not in stopwords and t not in mentions:
                mentions.append(t)

        return mentions

    def _interpret_llm(self, query: str) -> Optional[RetrievalSpec]:
        """
        LLM interpretation hook. If called, output MUST be strictly validated
        against approved schema and permitted intents.
        """
        # When LLM is not called or fails, returns None to allow deterministic fallback
        return None
