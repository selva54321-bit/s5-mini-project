"""
Integration tests for Neo4j Graph Storage against a live Neo4j database.
Automatically skipped if a live Neo4j instance is not reachable in the environment.
"""

import socket
import pytest
from app.config import settings
from app.graph.connection import db_manager
from app.graph.schema_init import initialize_schema
from app.graph.store import graph_store


def is_neo4j_live() -> bool:
    """Checks whether the configured Neo4j server is reachable on its TCP port."""
    try:
        # Extract host and port from bolt://host:port or neo4j://host:port
        uri_clean = settings.NEO4J_URI.split("://")[-1].split("@")[-1]
        host, port_str = uri_clean.split(":")
        port = int(port_str)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((host, port))
        sock.close()
        return bool(settings.NEO4J_PASSWORD)
    except Exception:
        return False


NEO4J_AVAILABLE = is_neo4j_live()


@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Live Neo4j instance is not reachable or NEO4J_PASSWORD is not set")
class TestNeo4jLiveIntegration:

    def test_live_health_check(self):
        health = db_manager.health_check()
        assert health["status"] == "HEALTHY"
        assert health["version"] != "unknown"

    def test_live_schema_initialization(self):
        report = initialize_schema()
        assert len(report["errors"]) == 0
        assert len(report["constraints_applied"]) > 0

    def test_live_node_crud_and_idempotency(self):
        test_urn = "urn:service:test:integration-test-service"
        # 1. Create node
        res1 = graph_store.upsert_node(
            label="Service",
            properties={"id": test_urn, "name": "Integration Test Service", "tier": "test"},
            provenance={"provenance_type": "MANUAL", "confidence": 1.0},
        )
        assert res1["id"] == test_urn

        # 2. Retrieve node
        fetched = graph_store.get_node(test_urn, label="Service")
        assert fetched is not None
        assert fetched["name"] == "Integration Test Service"

        # 3. Idempotent upsert with updated field
        res2 = graph_store.upsert_node(
            label="Service",
            properties={"id": test_urn, "tier": "test-v2", "env": "sandbox"},
        )
        assert res2["tier"] == "test-v2"
        assert res2["env"] == "sandbox"
        # Verify original name was preserved by += merge
        assert res2["name"] == "Integration Test Service"

        # 4. Clean up test node
        deleted = graph_store.delete_node(test_urn, label="Service")
        assert deleted is True
        assert graph_store.get_node(test_urn) is None
