"""
Unit tests for Neo4j Graph Storage Module (Phase 02).
Tests configuration, schema validation, driver lifecycle, schema initialization,
node/relationship operations, provenance preservation, transaction handling, and error cases using mocks.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.graph.schema import (
    NodeLabel,
    RelationshipType,
    ProvenanceType,
    validate_node_label,
    validate_relationship_type,
    validate_urn_identifier,
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATIONSHIPS,
    APPROVED_UNIQUENESS_CONSTRAINTS,
    APPROVED_LOOKUP_INDEXES,
    APPROVED_FULLTEXT_INDEXES,
)
from app.graph.connection import Neo4jConnectionManager, _mask_uri
from app.graph.schema_init import initialize_schema
from app.graph.store import GraphStore


# ── 1. Configuration & URI Masking Tests ──────────────────────────────────────

def test_uri_masking_with_credentials():
    masked = _mask_uri("bolt://admin:supersecret@localhost:7687")
    assert "supersecret" not in masked
    assert "admin" not in masked
    assert "***:***@" in masked


def test_uri_masking_without_credentials():
    masked = _mask_uri("bolt://localhost:7687")
    assert masked == "bolt://localhost:7687"


def test_connection_manager_missing_password_raises():
    mgr = Neo4jConnectionManager()
    mgr._driver = None  # Reset driver
    with patch("app.config.settings.NEO4J_PASSWORD", None):
        with pytest.raises(ValueError, match="NEO4J_PASSWORD environment variable"):
            mgr.connect()


# ── 2. Schema Validation & Injection Prevention Tests ─────────────────────────

def test_validate_node_label_valid():
    assert validate_node_label("Function") == "Function"
    assert validate_node_label("Service") == "Service"
    assert validate_node_label("SecurityRule") == "SecurityRule"


def test_validate_node_label_invalid_raises():
    with pytest.raises(ValueError, match="Invalid node label"):
        validate_node_label("UnknownLabel")

    with pytest.raises(ValueError, match="Invalid node label"):
        validate_node_label("Function; DROP DATABASE neo4j;")


def test_validate_relationship_type_valid():
    assert validate_relationship_type("CALLS") == "CALLS"
    assert validate_relationship_type("GOVERNED_BY") == "GOVERNED_BY"
    assert validate_relationship_type("calls") == "CALLS"  # Case-insensitive


def test_validate_relationship_type_invalid_raises():
    with pytest.raises(ValueError, match="Invalid relationship type"):
        validate_relationship_type("NON_EXISTENT_REL")

    with pytest.raises(ValueError, match="Invalid relationship type"):
        validate_relationship_type("CALLS]->(t) DELETE t //")


def test_validate_urn_identifier_valid():
    urn = "urn:func:kutty:app/agent/graph.py:rag_agent.invoke"
    assert validate_urn_identifier(urn) == urn


def test_validate_urn_identifier_invalid_raises():
    with pytest.raises(ValueError, match="does not match required URN format"):
        validate_urn_identifier("simple_func_name")

    with pytest.raises(ValueError, match="non-empty string"):
        validate_urn_identifier("")


# ── 3. Health Check & Connection Management Tests ─────────────────────────────

def test_health_check_healthy():
    mgr = Neo4jConnectionManager()
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_record = {"name": "Neo4j Kernel", "version": "5.26.0", "edition": "enterprise"}
    mock_session.run.return_value.single.return_value = mock_record

    mock_driver.session.return_value = mock_session
    mgr._driver = mock_driver

    health = mgr.health_check()
    assert health["status"] == "HEALTHY"
    assert health["version"] == "5.26.0"
    assert health["edition"] == "enterprise"
    assert "latency_ms" in health


def test_health_check_unhealthy():
    mgr = Neo4jConnectionManager()
    mock_driver = MagicMock()
    mock_driver.verify_connectivity.side_effect = Exception("Connection refused")
    mgr._driver = mock_driver

    health = mgr.health_check()
    assert health["status"] == "UNHEALTHY"
    assert "Connection refused" in health["error"]


def test_session_context_cleanup():
    mgr = Neo4jConnectionManager()
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session
    mgr._driver = mock_driver

    with mgr.session() as sess:
        assert sess == mock_session

    mock_session.close.assert_called_once()


# ── 4. Idempotent Schema Initialization Tests ─────────────────────────────────

def test_schema_initialization_statements_count():
    assert len(APPROVED_UNIQUENESS_CONSTRAINTS) == len(ALLOWED_NODE_LABELS)
    assert len(APPROVED_LOOKUP_INDEXES) > 0
    assert len(APPROVED_FULLTEXT_INDEXES) > 0


def test_schema_initialization_execution():
    mock_mgr = MagicMock()
    mock_session = MagicMock()
    mock_mgr.session.return_value.__enter__.return_value = mock_session

    with patch("app.graph.schema_init.db_manager", mock_mgr):
        report = initialize_schema()
        assert len(report["constraints_applied"]) == len(APPROVED_UNIQUENESS_CONSTRAINTS)
        assert len(report["lookup_indexes_applied"]) == len(APPROVED_LOOKUP_INDEXES)
        assert len(report["fulltext_indexes_applied"]) == len(APPROVED_FULLTEXT_INDEXES)
        assert len(report["errors"]) == 0

        # Verify that all applied statements used IF NOT EXISTS
        for stmt in report["constraints_applied"]:
            assert "IF NOT EXISTS" in stmt
        for stmt in report["lookup_indexes_applied"]:
            assert "IF NOT EXISTS" in stmt


# ── 5. Node Upsert & Idempotency Tests ─────────────────────────────────────────

def test_upsert_node_success():
    mock_mgr = MagicMock()
    store = GraphStore(manager=mock_mgr)

    # Setup execute_write mock to execute work_func
    def fake_execute_write(work_func):
        mock_tx = MagicMock()
        mock_record = {
            "n": {
                "id": "urn:service:kutty:auth-service",
                "name": "auth-service",
                "runtime": "python:3.12",
                "provenance_type": "STATIC_AST",
                "created_at": "2026-10-12T10:00:00Z",
                "updated_at": "2026-10-12T10:00:00Z",
            }
        }
        mock_tx.run.return_value.single.return_value = mock_record
        return work_func(mock_tx)

    mock_mgr.execute_write.side_effect = fake_execute_write

    result = store.upsert_node(
        label="Service",
        properties={"id": "urn:service:kutty:auth-service", "name": "auth-service", "runtime": "python:3.12"},
        provenance={"provenance_type": "STATIC_AST", "source_file_path": "app/config.py"},
    )

    assert result["id"] == "urn:service:kutty:auth-service"
    assert result["name"] == "auth-service"
    assert result["provenance_type"] == "STATIC_AST"


def test_upsert_node_missing_id_raises():
    store = GraphStore(manager=MagicMock())
    with pytest.raises(ValueError, match="must contain a valid 'id' URN"):
        store.upsert_node(label="Service", properties={"name": "auth-service"})


def test_upsert_node_invalid_label_raises():
    store = GraphStore(manager=MagicMock())
    with pytest.raises(ValueError, match="Invalid node label"):
        store.upsert_node(label="ArbitraryTable", properties={"id": "urn:service:kutty:auth-service"})


# ── 6. Relationship Upsert & Referential Integrity Tests ──────────────────────

def test_upsert_relationship_success():
    mock_mgr = MagicMock()
    store = GraphStore(manager=mock_mgr)

    def fake_execute_write(work_func):
        mock_tx = MagicMock()
        mock_record = {
            "type": "CALLS",
            "source_id": "urn:func:kutty:retriever:retrieve_node",
            "target_id": "urn:func:kutty:qdrant:search_enterprise_knowledge",
            "properties": {"line_number": 16, "provenance_type": "STATIC_AST"},
        }
        mock_tx.run.return_value.single.return_value = mock_record
        return work_func(mock_tx)

    mock_mgr.execute_write.side_effect = fake_execute_write

    result = store.upsert_relationship(
        rel_type="CALLS",
        source_id="urn:func:kutty:retriever:retrieve_node",
        target_id="urn:func:kutty:qdrant:search_enterprise_knowledge",
        properties={"line_number": 16},
        provenance={"provenance_type": "STATIC_AST"},
    )

    assert result["type"] == "CALLS"
    assert result["source_id"] == "urn:func:kutty:retriever:retrieve_node"
    assert result["target_id"] == "urn:func:kutty:qdrant:search_enterprise_knowledge"
    assert result["properties"]["line_number"] == 16


def test_upsert_relationship_missing_endpoints_raises():
    mock_mgr = MagicMock()
    store = GraphStore(manager=mock_mgr)

    def fake_execute_write(work_func):
        mock_tx = MagicMock()
        # Mock tx returning None because one or both endpoints don't match
        mock_tx.run.return_value.single.return_value = None
        return work_func(mock_tx)

    mock_mgr.execute_write.side_effect = fake_execute_write

    with pytest.raises(ValueError, match="One or both endpoints do not exist"):
        store.upsert_relationship(
            rel_type="CALLS",
            source_id="urn:func:kutty:missing_source",
            target_id="urn:func:kutty:missing_target",
        )


# ── 7. Explicit Deletion & Safety Tests ────────────────────────────────────────

def test_delete_node_explicit():
    mock_mgr = MagicMock()
    store = GraphStore(manager=mock_mgr)

    def fake_execute_write(work_func):
        mock_tx = MagicMock()
        mock_tx.run.return_value.single.return_value = {"deleted_count": 1}
        return work_func(mock_tx)

    mock_mgr.execute_write.side_effect = fake_execute_write

    deleted = store.delete_node("urn:service:kutty:auth-service", label="Service")
    assert deleted is True


def test_delete_node_invalid_urn_rejected():
    store = GraphStore(manager=MagicMock())
    with pytest.raises(ValueError, match="does not match required URN format"):
        store.delete_node("simple_id")


# ── 8. Batch Transaction Tests ────────────────────────────────────────────────

def test_upsert_nodes_batch():
    mock_mgr = MagicMock()
    store = GraphStore(manager=mock_mgr)

    def fake_execute_write(work_func):
        mock_tx = MagicMock()
        mock_tx.run.return_value.single.return_value = {"total": 2}
        return work_func(mock_tx)

    mock_mgr.execute_write.side_effect = fake_execute_write

    batch = [
        {"id": "urn:func:kutty:f1", "name": "f1"},
        {"id": "urn:func:kutty:f2", "name": "f2"},
    ]
    total = store.upsert_nodes_batch(label="Function", nodes_data=batch)
    assert total == 2
