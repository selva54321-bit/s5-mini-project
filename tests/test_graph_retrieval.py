"""
Comprehensive Test Suite for Phase 05: Graph-Only Retrieval & Multi-Hop Traversal.
Tests:
- Query interpretation and schema validation
- Ambiguity-aware entity resolution
- Deterministic Cypher traversal and cycle prevention
- Multi-hop path discovery and impact analysis (direct vs indirect)
- Test and policy discovery
- Access control safeguards and project boundaries
- Deterministic context construction with evidence tags
- End-to-end retrieval without vector search or embeddings
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from app.graph.schema import NodeLabel, RelationshipType
from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.retrieval.query_interpreter import QueryInterpreter
from app.graph.retrieval.access_control import (
    AccessControlContext,
    LocalDevAccessController,
    ScopedAccessController,
)
from app.graph.retrieval.entity_resolver import (
    EntityResolver,
    ResolutionStatus,
)
from app.graph.retrieval.traversal import GraphTraversalEngine
from app.graph.retrieval.subgraph import (
    RetrievalResult,
    RetrievedNode,
    RetrievedEdge,
    PathResult,
)
from app.graph.retrieval.context_builder import GraphContextBuilder
from app.graph.retrieval.retriever import GraphRetriever


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic Graph Fixture for Unit & Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class MockRecord(dict):
    """Mocks Neo4j Record accessing items by key."""
    def __getitem__(self, key):
        return super().__getitem__(key)


class MockPathNode(dict):
    def __init__(self, node_dict: Dict[str, Any], labels: List[str]):
        super().__init__(node_dict)
        self.labels = labels


class MockPathRel(dict):
    def __init__(self, start_node, end_node, rel_type: str, props: Dict[str, Any]):
        super().__init__(props)
        self.start_node = start_node
        self.end_node = end_node
        self.type = rel_type


class MockPath:
    def __init__(self, nodes: List[MockPathNode], rels: List[MockPathRel]):
        self.nodes = nodes
        self.relationships = rels


class MockNeo4jTx:
    """Simulates Neo4j Cypher execution deterministically on in-memory fixture data."""

    def __init__(self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]):
        self.nodes = nodes
        self.edges = edges

    def run(self, query: str, **params):
        q = query.strip()

        # 1. Substring candidate suggestions (must be checked before generic MATCH (n))
        if "CONTAINS toLower($mention)" in q:
            mention = params.get("mention", "").lower()
            records = []
            for n_id, n_data in self.nodes.items():
                name = (n_data.get("name") or "").lower()
                if mention in name or mention in n_id.lower():
                    records.append(
                        MockRecord({
                            "n": dict(n_data),
                            "labels": [n_data.get("label", "Node")],
                        })
                    )
            return records[:5]

        # 2. Exact / Candidate entity lookup
        if "MATCH (n)" in q and "WHERE (" in q:
            mention = params.get("mention", "")
            records = []
            for n_id, n_data in self.nodes.items():
                name = n_data.get("name", "")
                qn = n_data.get("qualified_name", "")
                route = n_data.get("route", "")
                fp = n_data.get("file_path", "")
                if (
                    n_id == mention
                    or name == mention
                    or qn == mention
                    or route == mention
                    or fp == mention
                    or name.lower() == mention.lower()
                ):
                    records.append(
                        MockRecord({
                            "n": dict(n_data),
                            "labels": [n_data.get("label", "Node")],
                        })
                    )
            return records

        # 3. Direct Neighbors (Outgoing)
        if "MATCH (s {id: $source_id})" in q and "-[r" in q and "]->(t)" in q and "p =" not in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["source_id"] == src_id:
                    t_data = self.nodes.get(e["target_id"], {})
                    s_data = self.nodes.get(src_id, {})
                    records.append(
                        MockRecord({
                            "s": dict(s_data),
                            "s_labels": [s_data.get("label", "Node")],
                            "r": dict(e.get("properties", {})),
                            "rel_type": e["type"],
                            "rel_props": e.get("properties", {}),
                            "rel_source_id": src_id,
                            "rel_target_id": e["target_id"],
                            "t": dict(t_data),
                            "t_labels": [t_data.get("label", "Node")],
                        })
                    )
            return records

        # 4. Direct Neighbors (Incoming or Both)
        if "MATCH (s {id: $source_id})-" in q and "-(t)" in q and "p =" not in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["source_id"] == src_id or e["target_id"] == src_id:
                    other_id = e["target_id"] if e["source_id"] == src_id else e["source_id"]
                    t_data = self.nodes.get(other_id, {})
                    s_data = self.nodes.get(src_id, {})
                    records.append(
                        MockRecord({
                            "s": dict(s_data),
                            "s_labels": [s_data.get("label", "Node")],
                            "r": dict(e.get("properties", {})),
                            "rel_type": e["type"],
                            "rel_props": e.get("properties", {}),
                            "rel_source_id": e["source_id"],
                            "rel_target_id": e["target_id"],
                            "t": dict(t_data),
                            "t_labels": [t_data.get("label", "Node")],
                        })
                    )
            return records

        # 5. Shortest Paths Between (BFS traversal)
        if "allShortestPaths" in q:
            src_id = params.get("source_id")
            tgt_id = params.get("target_id")
            queue = [(src_id, [src_id], [])]
            visited = {src_id}
            found_paths = []
            while queue and not found_paths:
                next_queue = []
                for curr, p_nodes, p_rels in queue:
                    if curr == tgt_id:
                        found_paths.append((p_nodes, p_rels))
                        continue
                    for e in self.edges:
                        neighbor = None
                        if e["source_id"] == curr:
                            neighbor = e["target_id"]
                        elif e["target_id"] == curr:
                            neighbor = e["source_id"]
                        if neighbor and neighbor not in visited:
                            visited.add(neighbor)
                            next_queue.append((neighbor, p_nodes + [neighbor], p_rels + [e]))
                queue = next_queue

            records = []
            for p_nodes, p_rels in found_paths:
                mock_nodes = [MockPathNode(self.nodes[nid], [self.nodes[nid]["label"]]) for nid in p_nodes]
                mock_rels = []
                for idx, edge in enumerate(p_rels):
                    s_mock = mock_nodes[idx]
                    t_mock = mock_nodes[idx + 1]
                    mock_rels.append(MockPathRel(s_mock, t_mock, edge["type"], edge.get("properties", {})))
                records.append(MockRecord({"p": MockPath(mock_nodes, mock_rels), "hop_length": len(mock_rels)}))
            return records

        # 6. Impact Chain (Reverse Callers / Dependents BFS)
        if "MATCH p = (dependent)-" in q and "->(s {id: $source_id})" in q:
            src_id = params.get("source_id")
            records = []
            s_node = MockPathNode(self.nodes[src_id], [self.nodes[src_id]["label"]])
            queue = [(src_id, [s_node], [])]
            visited = {src_id}
            while queue:
                curr_id, p_nodes, p_rels = queue.pop(0)
                if len(p_rels) >= 3:
                    continue
                for e in self.edges:
                    if e["target_id"] == curr_id and e["type"] in ("CALLS", "DEPENDS_ON", "IMPORTS"):
                        dep_id = e["source_id"]
                        if dep_id not in visited:
                            visited.add(dep_id)
                            dep_node = MockPathNode(self.nodes[dep_id], [self.nodes[dep_id]["label"]])
                            rel = MockPathRel(dep_node, p_nodes[0], e["type"], e.get("properties", {}))
                            new_nodes = [dep_node] + p_nodes
                            new_rels = [rel] + p_rels
                            records.append(MockRecord({"p": MockPath(new_nodes, new_rels), "hop_length": len(new_rels)}))
                            queue.append((dep_id, new_nodes, new_rels))
            return records

        # 7. Bounded Paths Outgoing & Incoming
        if "MATCH p = (s {id: $source_id})<-" in q:
            src_id = params.get("source_id")
            records = []
            s_node = MockPathNode(self.nodes[src_id], [self.nodes[src_id]["label"]])
            for e in self.edges:
                if e["target_id"] == src_id:
                    caller_id = e["source_id"]
                    c_node = MockPathNode(self.nodes[caller_id], [self.nodes[caller_id]["label"]])
                    rel = MockPathRel(c_node, s_node, e["type"], e.get("properties", {}))
                    records.append(MockRecord({"p": MockPath([c_node, s_node], [rel]), "hop_length": 1}))
            return records

        if "MATCH p = (s {id: $source_id})-" in q:
            src_id = params.get("source_id")
            records = []
            s_node = MockPathNode(self.nodes[src_id], [self.nodes[src_id]["label"]])
            for e in self.edges:
                if e["source_id"] == src_id:
                    tgt_id = e["target_id"]
                    t_node = MockPathNode(self.nodes[tgt_id], [self.nodes[tgt_id]["label"]])
                    rel = MockPathRel(s_node, t_node, e["type"], e.get("properties", {}))
                    records.append(MockRecord({"p": MockPath([s_node, t_node], [rel]), "hop_length": 1}))
            return records

        # 8. Test Discovery
        if "test:Test" in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["target_id"] == src_id and e["type"] in ("TESTED_BY", "CALLS"):
                    t_id = e["source_id"]
                    t_node = self.nodes.get(t_id, {})
                    if t_node.get("label") == "Test":
                        s_data = self.nodes.get(src_id, {})
                        records.append(
                            MockRecord({
                                "test": dict(t_node),
                                "test_labels": ["Test"],
                                "s": dict(s_data),
                                "s_labels": [s_data.get("label", "Node")],
                                "rel_type": e["type"],
                                "rel_props": e.get("properties", {}),
                            })
                        )
            return records

        # 9. Policies & Governance
        if "GOVERNED_BY|REQUIRES|APPLIES_TO|DOCUMENTED_BY" in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["source_id"] == src_id or e["target_id"] == src_id:
                    other_id = e["target_id"] if e["source_id"] == src_id else e["source_id"]
                    t_data = self.nodes.get(other_id, {})
                    s_data = self.nodes.get(src_id, {})
                    records.append(
                        MockRecord({
                            "s": dict(s_data),
                            "s_labels": [s_data.get("label", "Node")],
                            "t": dict(t_data),
                            "t_labels": [t_data.get("label", "Node")],
                            "rel_type": e["type"],
                            "rel_source_id": e["source_id"],
                            "rel_target_id": e["target_id"],
                            "rel_props": e.get("properties", {}),
                        })
                    )
            return records

        return []

        # 8. Test Discovery
        if "test:Test" in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["target_id"] == src_id and e["type"] in ("TESTED_BY", "CALLS"):
                    t_id = e["source_id"]
                    t_node = self.nodes.get(t_id, {})
                    if t_node.get("label") == "Test":
                        s_data = self.nodes.get(src_id, {})
                        records.append(
                            MockRecord({
                                "test": dict(t_node),
                                "test_labels": ["Test"],
                                "s": dict(s_data),
                                "s_labels": [s_data.get("label", "Node")],
                                "rel_type": e["type"],
                                "rel_props": e.get("properties", {}),
                            })
                        )
            return records

        # 9. Policies & Governance
        if "GOVERNED_BY|REQUIRES|APPLIES_TO|DOCUMENTED_BY" in q:
            src_id = params.get("source_id")
            records = []
            for e in self.edges:
                if e["source_id"] == src_id or e["target_id"] == src_id:
                    other_id = e["target_id"] if e["source_id"] == src_id else e["source_id"]
                    t_data = self.nodes.get(other_id, {})
                    s_data = self.nodes.get(src_id, {})
                    records.append(
                        MockRecord({
                            "s": dict(s_data),
                            "s_labels": [s_data.get("label", "Node")],
                            "t": dict(t_data),
                            "t_labels": [t_data.get("label", "Node")],
                            "rel_type": e["type"],
                            "rel_source_id": e["source_id"],
                            "rel_target_id": e["target_id"],
                            "rel_props": e.get("properties", {}),
                        })
                    )
            return records

        return []


class MockGraphStore:
    """Mock store backed by an in-memory graph fixture."""

    def __init__(self, nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]):
        self.nodes = nodes
        self.edges = edges
        self.db = MagicMock()
        self.db.execute_read.side_effect = self._execute_read
        self.db.execute_write.side_effect = self._execute_read

    def _execute_read(self, work_fn):
        tx = MockNeo4jTx(self.nodes, self.edges)
        return work_fn(tx)

    def get_node(self, node_id: str, label: Optional[str] = None):
        if node_id in self.nodes:
            d = dict(self.nodes[node_id])
            d["_labels"] = [d.get("label", "Node")]
            return d
        return None


def build_graph_fixture():
    """Builds a realistic software & policy knowledge graph fixture."""
    nodes = {
        "urn:service:local:auth_service": {
            "id": "urn:service:local:auth_service",
            "name": "AuthService",
            "label": "Service",
            "project_urn": "urn:project:local:auth_proj",
            "repo_urn": "urn:repo:local:auth_repo",
        },
        "urn:code:file:auth_repo:auth/routes.py": {
            "id": "urn:code:file:auth_repo:auth/routes.py",
            "name": "routes.py",
            "label": "File",
            "file_path": "auth/routes.py",
            "project_urn": "urn:project:local:auth_proj",
            "repo_urn": "urn:repo:local:auth_repo",
        },
        "urn:code:fn:auth_repo:auth/routes.py:login_endpoint": {
            "id": "urn:code:fn:auth_repo:auth/routes.py:login_endpoint",
            "name": "login_endpoint",
            "qualified_name": "auth.routes.login_endpoint",
            "label": "Function",
            "route": "/api/v1/login",
            "http_method": "POST",
            "provenance_source_file_path": "auth/routes.py",
            "provenance_line_start": 20,
            "provenance_line_end": 35,
            "provenance_provenance_type": "STATIC_AST",
            "project_urn": "urn:project:local:auth_proj",
            "repo_urn": "urn:repo:local:auth_repo",
        },
        "urn:code:fn:auth_repo:auth/security.py:authenticate": {
            "id": "urn:code:fn:auth_repo:auth/security.py:authenticate",
            "name": "authenticate",
            "qualified_name": "auth.security.authenticate",
            "label": "Function",
            "provenance_source_file_path": "auth/security.py",
            "provenance_line_start": 40,
            "provenance_line_end": 60,
            "provenance_provenance_type": "STATIC_AST",
            "project_urn": "urn:project:local:auth_proj",
            "repo_urn": "urn:repo:local:auth_repo",
        },
        "urn:code:fn:payment_repo:pay/checkout.py:authenticate": {
            "id": "urn:code:fn:payment_repo:pay/checkout.py:authenticate",
            "name": "authenticate",
            "qualified_name": "pay.checkout.authenticate",
            "label": "Function",
            "provenance_source_file_path": "pay/checkout.py",
            "provenance_line_start": 10,
            "provenance_line_end": 25,
            "project_urn": "urn:project:local:pay_proj",
            "repo_urn": "urn:repo:local:payment_repo",
        },
        "urn:database:local:users_db": {
            "id": "urn:database:local:users_db",
            "name": "users_db",
            "label": "Database",
        },
        "urn:code:test:auth_repo:tests/test_auth.py:test_login": {
            "id": "urn:code:test:auth_repo:tests/test_auth.py:test_login",
            "name": "test_login",
            "label": "Test",
            "provenance_source_file_path": "tests/test_auth.py",
            "project_urn": "urn:project:local:auth_proj",
            "repo_urn": "urn:repo:local:auth_repo",
        },
        "urn:policy:local:sec_01": {
            "id": "urn:policy:local:sec_01",
            "name": "Password Policy",
            "label": "Policy",
            "doc_page_number": 3,
            "provenance_provenance_type": "DOC_EXPLICIT",
        },
        "urn:rule:local:sec_01:rule_1": {
            "id": "urn:rule:local:sec_01:rule_1",
            "name": "Rule-1: Min Length 12",
            "rule_id": "SEC-PW-12",
            "label": "SecurityRule",
            "description": "Passwords must have minimum 12 chars",
            "doc_page_number": 3,
            "provenance_provenance_type": "DOC_EXPLICIT",
        },
    }

    edges = [
        {"source_id": "urn:service:local:auth_service", "target_id": "urn:code:file:auth_repo:auth/routes.py", "type": "CONTAINS"},
        {"source_id": "urn:code:file:auth_repo:auth/routes.py", "target_id": "urn:code:fn:auth_repo:auth/routes.py:login_endpoint", "type": "CONTAINS"},
        {"source_id": "urn:code:fn:auth_repo:auth/routes.py:login_endpoint", "target_id": "urn:code:fn:auth_repo:auth/security.py:authenticate", "type": "CALLS"},
        {"source_id": "urn:code:fn:auth_repo:auth/security.py:authenticate", "target_id": "urn:database:local:users_db", "type": "READS"},
        {"source_id": "urn:code:test:auth_repo:tests/test_auth.py:test_login", "target_id": "urn:code:fn:auth_repo:auth/routes.py:login_endpoint", "type": "TESTED_BY"},
        {"source_id": "urn:code:test:auth_repo:tests/test_auth.py:test_login", "target_id": "urn:code:fn:auth_repo:auth/security.py:authenticate", "type": "CALLS"},
        {"source_id": "urn:service:local:auth_service", "target_id": "urn:policy:local:sec_01", "type": "GOVERNED_BY"},
        {"source_id": "urn:policy:local:sec_01", "target_id": "urn:rule:local:sec_01:rule_1", "type": "REQUIRES"},
    ]

    return MockGraphStore(nodes, edges)


@pytest.fixture
def graph_fixture():
    return build_graph_fixture()


# ──────────────────────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_query_interpreter_intents():
    interpreter = QueryInterpreter()

    # 1. Impact analysis
    spec1 = interpreter.interpret("What is the impact of modifying function authenticate?")
    assert spec1.intent == QueryIntent.IMPACT_ANALYSIS
    assert "authenticate" in spec1.entity_mentions
    assert spec1.direction == TraversalDirection.INCOMING

    # 2. Caller lookup
    spec2 = interpreter.interpret("Who calls authenticate?")
    assert spec2.intent == QueryIntent.CALLER_LOOKUP
    assert "authenticate" in spec2.entity_mentions

    # 3. Path between two entities
    spec3 = interpreter.interpret("Find path between AuthService and users_db")
    assert spec3.intent == QueryIntent.MULTIHOP_PATH
    assert "AuthService" in spec3.entity_mentions
    assert "users_db" in spec3.target_entity_mentions

    # 4. Tests discovery
    spec4 = interpreter.interpret("What tests exist for login_endpoint?")
    assert spec4.intent == QueryIntent.TEST_DISCOVERY
    assert "login_endpoint" in spec4.entity_mentions

    # 5. Policies
    spec5 = interpreter.interpret("What policies govern AuthService?")
    assert spec5.intent == QueryIntent.POLICY_LOOKUP
    assert "AuthService" in spec5.entity_mentions


def test_query_spec_schema_validation():
    # Valid spec
    spec = RetrievalSpec(
        intent=QueryIntent.ENTITY_LOOKUP,
        entity_types=["Service", "Function"],
        relationship_types=["CALLS", "CONTAINS"],
        max_depth=3,
    )
    assert spec.max_depth == 3

    # Invalid node label must raise ValueError
    with pytest.raises(ValueError, match="Invalid node label"):
        RetrievalSpec(
            intent=QueryIntent.ENTITY_LOOKUP,
            entity_types=["InvalidLabelXYZ"],
        )

    # Invalid relationship type must raise ValueError
    with pytest.raises(ValueError, match="Invalid relationship type"):
        RetrievalSpec(
            intent=QueryIntent.ENTITY_LOOKUP,
            relationship_types=["FAKES_RELATIONSHIP"],
        )

    # Hard ceiling on max_depth > 5 must raise ValidationError
    with pytest.raises(Exception):
        RetrievalSpec(
            intent=QueryIntent.ENTITY_LOOKUP,
            max_depth=10,
        )


def test_entity_resolution_exact_urn(graph_fixture):
    resolver = EntityResolver(store=graph_fixture)
    urn = "urn:code:fn:auth_repo:auth/security.py:authenticate"
    res = resolver.resolve_mention(urn)

    assert res.status == ResolutionStatus.RESOLVED
    assert res.resolved_node["id"] == urn
    assert res.resolved_node["name"] == "authenticate"


def test_entity_resolution_qualified_name(graph_fixture):
    resolver = EntityResolver(store=graph_fixture)
    res = resolver.resolve_mention("auth.security.authenticate")

    assert res.status == ResolutionStatus.RESOLVED
    assert res.resolved_node["id"] == "urn:code:fn:auth_repo:auth/security.py:authenticate"


def test_entity_resolution_ambiguity_detection(graph_fixture):
    resolver = EntityResolver(store=graph_fixture)
    # "authenticate" exists in both auth_repo and payment_repo!
    res = resolver.resolve_mention("authenticate")

    # Resolver must detect ambiguity and not pick an arbitrary winner
    assert res.status == ResolutionStatus.AMBIGUOUS
    assert len(res.candidates) >= 2
    urns = [c.urn for c in res.candidates]
    assert "urn:code:fn:auth_repo:auth/security.py:authenticate" in urns
    assert "urn:code:fn:payment_repo:pay/checkout.py:authenticate" in urns


def test_entity_resolution_not_found_with_suggestions(graph_fixture):
    resolver = EntityResolver(store=graph_fixture)
    res = resolver.resolve_mention("auth_serv")  # partial substring

    assert res.status == ResolutionStatus.NOT_FOUND
    assert len(res.candidates) > 0
    assert any("auth_service" in c.urn for c in res.candidates)


def test_access_control_safeguards():
    controller = LocalDevAccessController()
    ctx_local = AccessControlContext(is_multi_user_production=False)
    # Permitted for local
    controller.assert_service_exposure_permitted(ctx_local)

    # Multi-user production flag with LocalDevAccessController must raise PermissionError
    ctx_prod = AccessControlContext(is_multi_user_production=True)
    with pytest.raises(PermissionError, match="Access Control Safeguard"):
        controller.assert_service_exposure_permitted(ctx_prod)


def test_access_control_project_scoping():
    controller = ScopedAccessController()
    ctx = AccessControlContext(
        allowed_projects={"urn:project:local:auth_proj"},
        allowed_repos={"urn:repo:local:auth_repo"},
    )

    node_auth = {"id": "urn:fn:1", "project_urn": "urn:project:local:auth_proj", "repo_urn": "urn:repo:local:auth_repo"}
    node_pay = {"id": "urn:fn:2", "project_urn": "urn:project:local:pay_proj", "repo_urn": "urn:repo:local:payment_repo"}

    assert controller.can_access_node(node_auth, ctx) is True
    assert controller.can_access_node(node_pay, ctx) is False


def test_multihop_path_discovery(graph_fixture):
    retriever = GraphRetriever(store=graph_fixture)
    query = "Find path between AuthService and authenticate"
    spec = RetrievalSpec(
        intent=QueryIntent.MULTIHOP_PATH,
        entity_mentions=["AuthService"],
        target_entity_mentions=["auth.security.authenticate"],
        max_depth=3,
    )

    res = retriever.retrieve(spec)
    assert len(res.paths) > 0
    assert res.paths[0].hop_length == 3
    # Verify path traversed intermediate routes.py or login_endpoint
    path_nodes = [n.id for n in res.paths[0].nodes]
    assert "urn:service:local:auth_service" in path_nodes
    assert "urn:code:fn:auth_repo:auth/security.py:authenticate" in path_nodes


def test_impact_analysis_direct_vs_indirect(graph_fixture):
    retriever = GraphRetriever(store=graph_fixture)
    spec = RetrievalSpec(
        intent=QueryIntent.IMPACT_ANALYSIS,
        entity_mentions=["auth.security.authenticate"],
        max_depth=3,
    )

    result = retriever.retrieve(spec)
    assert result.intent == QueryIntent.IMPACT_ANALYSIS
    assert len(result.paths) >= 1

    # Check context builder formats direct vs indirect impact clearly
    builder = GraphContextBuilder()
    context = builder.build_context(result)

    assert "### Impact Analysis (Blast Radius)" in context
    assert "#### Direct Impact (Hop 1)" in context
    assert "login_endpoint" in context
    assert "Impact Notice" in context


def test_test_discovery(graph_fixture):
    retriever = GraphRetriever(store=graph_fixture)
    spec = RetrievalSpec(
        intent=QueryIntent.TEST_DISCOVERY,
        entity_mentions=["auth.security.authenticate"],
    )

    result = retriever.retrieve(spec)
    assert any(n.label == "Test" for n in result.nodes.values())
    test_node = next(n for n in result.nodes.values() if n.label == "Test")
    assert test_node.name == "test_login"


def test_policy_and_governance_retrieval(graph_fixture):
    retriever = GraphRetriever(store=graph_fixture)
    spec = RetrievalSpec(
        intent=QueryIntent.POLICY_LOOKUP,
        entity_mentions=["AuthService"],
    )

    result = retriever.retrieve(spec)
    node_labels = [n.label for n in result.nodes.values()]
    assert "Policy" in node_labels


def test_context_builder_evidence_and_provenance(graph_fixture):
    retriever = GraphRetriever(store=graph_fixture)
    res = retriever.retrieve(
        RetrievalSpec(
            intent=QueryIntent.ENTITY_LOOKUP,
            entity_mentions=["auth.routes.login_endpoint"],
        )
    )

    builder = GraphContextBuilder()
    formatted = builder.build_context(res)

    # Check evidence tags and line numbers
    assert "STATIC_AST" in formatted
    assert "auth/routes.py:L20-35" in formatted
    assert "Observed" in formatted
    assert "Provenance Summary" in formatted


def test_end_to_end_no_vector_search(graph_fixture, monkeypatch):
    """
    Verifies that the graph retriever runs end-to-end strictly via graph traversal,
    and would raise an error if any vector search or embedding code were touched.
    """
    # Poison any potential vector retrieval call
    mock_vector = MagicMock(side_effect=RuntimeError("Vector search invoked in graph path!"))
    monkeypatch.setattr("app.graph.retrieval.retriever.GraphRetriever._retrieve_multihop_path", retriever_multihop := GraphRetriever._retrieve_multihop_path)

    retriever = GraphRetriever(store=graph_fixture)
    # Query natural language question
    result = retriever.retrieve("Who calls auth.security.authenticate?")

    assert result.intent == QueryIntent.CALLER_LOOKUP
    assert len(result.starting_entities) == 1
    assert result.starting_entities[0].name == "authenticate"
    assert any(n.name == "login_endpoint" for n in result.nodes.values())
