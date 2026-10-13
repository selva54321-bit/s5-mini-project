"""
Enterprise Graph Storage Operations Module.
Implements parameterized Cypher node and relationship CRUD, batch transactions,
whitelist validation against Cypher injection, and strict identity/provenance enforcement.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from app.graph.connection import db_manager, Neo4jConnectionManager
from app.graph.schema import (
    validate_node_label,
    validate_relationship_type,
    validate_urn_identifier,
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATIONSHIPS,
)

logger = logging.getLogger("graph.store")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class GraphStore:
    """
    High-level repository for executing safe, parameterized Graph operations.
    """

    def __init__(self, manager: Optional[Neo4jConnectionManager] = None):
        self.db = manager or db_manager

    # ──────────────────────────────────────────────────────────────────────────
    # Node Operations
    # ──────────────────────────────────────────────────────────────────────────

    def upsert_node(
        self,
        label: str,
        properties: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Creates or updates a node identified by its stable URN 'id'.
        Idempotent: merges on 'id' and updates attributes without nullifying unpassed properties.
        """
        valid_label = validate_node_label(label)
        node_id = properties.get("id")
        if not node_id:
            raise ValueError(f"Node properties must contain a valid 'id' URN. Received: {properties}")
        valid_id = validate_urn_identifier(str(node_id))

        props = dict(properties)
        props["id"] = valid_id

        # Merge provenance metadata into node properties if provided
        if provenance:
            for k, v in provenance.items():
                if v is not None:
                    props[f"provenance_{k}" if not k.startswith("provenance_") else k] = v

        now_iso = datetime.now(timezone.utc).isoformat()

        cypher = f"""
        MERGE (n:{valid_label} {{id: $id}})
        ON CREATE SET n = $props, n.created_at = $now_iso
        ON MATCH SET n += $props, n.updated_at = $now_iso
        RETURN n
        """

        def _work(tx):
            res = tx.run(cypher, id=valid_id, props=props, now_iso=now_iso)
            record = res.single()
            return dict(record["n"]) if record else {}

        return self.db.execute_write(_work)

    def get_node(self, node_id: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves a node by its stable URN identifier.
        """
        valid_id = validate_urn_identifier(node_id)
        if label:
            valid_label = validate_node_label(label)
            cypher = f"MATCH (n:{valid_label} {{id: $id}}) RETURN n, labels(n) AS labels"
        else:
            cypher = "MATCH (n {id: $id}) RETURN n, labels(n) AS labels"

        def _work(tx):
            res = tx.run(cypher, id=valid_id)
            record = res.single()
            if not record:
                return None
            data = dict(record["n"])
            data["_labels"] = record["labels"]
            return data

        return self.db.execute_read(_work)

    def list_nodes_by_label(self, label: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns a bounded collection of nodes for an approved label."""
        valid_label = validate_node_label(label)
        bounded_limit = max(1, min(limit, 200))
        cypher = f"MATCH (n:{valid_label}) RETURN n, labels(n) AS labels ORDER BY n.name, n.id LIMIT $limit"

        def _work(tx):
            return [
                {**dict(record["n"]), "_labels": record["labels"]}
                for record in tx.run(cypher, limit=bounded_limit)
            ]

        return self.db.execute_read(_work)

    def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        label: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Updates permitted properties on an existing node without overwriting omitted fields.
        Protected property 'id' cannot be altered.
        """
        valid_id = validate_urn_identifier(node_id)
        props_to_update = {k: v for k, v in properties.items() if k != "id"}

        if label:
            valid_label = validate_node_label(label)
            cypher = f"""
            MATCH (n:{valid_label} {{id: $id}})
            SET n += $props, n.updated_at = $now_iso
            RETURN n, labels(n) AS labels
            """
        else:
            cypher = """
            MATCH (n {id: $id})
            SET n += $props, n.updated_at = $now_iso
            RETURN n, labels(n) AS labels
            """

        now_iso = datetime.now(timezone.utc).isoformat()

        def _work(tx):
            res = tx.run(cypher, id=valid_id, props=props_to_update, now_iso=now_iso)
            record = res.single()
            if not record:
                return None
            data = dict(record["n"])
            data["_labels"] = record["labels"]
            return data

        return self.db.execute_write(_work)

    def delete_node(
        self,
        node_id: str,
        label: Optional[str] = None,
        detach: bool = True,
    ) -> bool:
        """
        Explicitly deletes a single node by its stable identifier.
        Requires explicit node_id; cannot wipe or wildcard.
        """
        valid_id = validate_urn_identifier(node_id)
        detach_clause = "DETACH " if detach else ""

        if label:
            valid_label = validate_node_label(label)
            cypher = f"MATCH (n:{valid_label} {{id: $id}}) {detach_clause}DELETE n RETURN count(n) AS deleted_count"
        else:
            cypher = f"MATCH (n {{id: $id}}) {detach_clause}DELETE n RETURN count(n) AS deleted_count"

        def _work(tx):
            res = tx.run(cypher, id=valid_id)
            record = res.single()
            return bool(record and record["deleted_count"] > 0)

        return self.db.execute_write(_work)

    # ──────────────────────────────────────────────────────────────────────────
    # Relationship Operations
    # ──────────────────────────────────────────────────────────────────────────

    def upsert_relationship(
        self,
        rel_type: str,
        source_id: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Creates or updates a typed directed edge between source_id and target_id.
        Preserves provenance metadata and enforces referential integrity.
        """
        valid_type = validate_relationship_type(rel_type)
        valid_src = validate_urn_identifier(source_id)
        valid_tgt = validate_urn_identifier(target_id)

        props = dict(properties or {})
        if provenance:
            for k, v in provenance.items():
                if v is not None:
                    props[f"provenance_{k}" if not k.startswith("provenance_") else k] = v

        now_iso = datetime.now(timezone.utc).isoformat()

        # Explicitly verify that both endpoints exist before attempting to merge relationship
        cypher = f"""
        MATCH (s {{id: $src_id}})
        MATCH (t {{id: $tgt_id}})
        MERGE (s)-[r:{valid_type}]->(t)
        ON CREATE SET r = $props, r.created_at = $now_iso
        ON MATCH SET r += $props, r.updated_at = $now_iso
        RETURN type(r) AS type, properties(r) AS properties, s.id AS source_id, t.id AS target_id
        """

        def _work(tx):
            res = tx.run(cypher, src_id=valid_src, tgt_id=valid_tgt, props=props, now_iso=now_iso)
            record = res.single()
            if not record:
                raise ValueError(
                    f"Cannot create relationship '{valid_type}': One or both endpoints "
                    f"do not exist (source='{valid_src}', target='{valid_tgt}')."
                )
            return {
                "type": record["type"],
                "source_id": record["source_id"],
                "target_id": record["target_id"],
                "properties": dict(record["properties"]),
            }

        return self.db.execute_write(_work)

    def get_relationships(
        self,
        node_id: str,
        direction: str = "BOTH",
        rel_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relationships connected to a node.
        direction can be 'OUT', 'IN', or 'BOTH'.
        """
        valid_id = validate_urn_identifier(node_id)
        type_clause = f":{validate_relationship_type(rel_type)}" if rel_type else ""

        if direction.upper() == "OUT":
            cypher = f"""
            MATCH (s {{id: $id}})-[r{type_clause}]->(t)
            RETURN type(r) AS type, properties(r) AS properties, s.id AS source_id, t.id AS target_id, labels(t) AS target_labels
            """
        elif direction.upper() == "IN":
            cypher = f"""
            MATCH (s)-[r{type_clause}]->(t {{id: $id}})
            RETURN type(r) AS type, properties(r) AS properties, s.id AS source_id, t.id AS target_id, labels(s) AS source_labels
            """
        else:
            cypher = f"""
            MATCH (s {{id: $id}})-[r{type_clause}]-(t)
            RETURN type(r) AS type, properties(r) AS properties, startNode(r).id AS source_id, endNode(r).id AS target_id, labels(t) AS other_labels
            """

        def _work(tx):
            res = tx.run(cypher, id=valid_id)
            return [
                {
                    "type": record["type"],
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "properties": dict(record["properties"]),
                }
                for record in res
            ]

        return self.db.execute_read(_work)

    def delete_relationship(
        self,
        rel_type: str,
        source_id: str,
        target_id: str,
    ) -> bool:
        """
        Explicitly removes a typed directed relationship between two nodes.
        """
        valid_type = validate_relationship_type(rel_type)
        valid_src = validate_urn_identifier(source_id)
        valid_tgt = validate_urn_identifier(target_id)

        cypher = f"""
        MATCH (s {{id: $src_id}})-[r:{valid_type}]->(t {{id: $tgt_id}})
        DELETE r
        RETURN count(r) AS deleted_count
        """

        def _work(tx):
            res = tx.run(cypher, src_id=valid_src, tgt_id=valid_tgt)
            record = res.single()
            return bool(record and record["deleted_count"] > 0)

        return self.db.execute_write(_work)

    # ──────────────────────────────────────────────────────────────────────────
    # Transactional Batch Operations
    # ──────────────────────────────────────────────────────────────────────────

    def upsert_nodes_batch(
        self,
        label: str,
        nodes_data: List[Dict[str, Any]],
    ) -> int:
        """
        Transactional bulk upsert of nodes using UNWIND.
        Validates all IDs and labels before execution.
        """
        if not nodes_data:
            return 0

        valid_label = validate_node_label(label)
        clean_batch = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in nodes_data:
            node_id = item.get("id")
            if not node_id:
                raise ValueError(f"Every node in batch must contain an 'id'. Received: {item}")
            clean_item = dict(item)
            clean_item["id"] = validate_urn_identifier(str(node_id))
            clean_batch.append(clean_item)

        cypher = f"""
        UNWIND $batch AS item
        MERGE (n:{valid_label} {{id: item.id}})
        ON CREATE SET n = item, n.created_at = $now_iso
        ON MATCH SET n += item, n.updated_at = $now_iso
        RETURN count(n) AS total
        """

        def _work(tx):
            res = tx.run(cypher, batch=clean_batch, now_iso=now_iso)
            record = res.single()
            return record["total"] if record else 0

        return self.db.execute_write(_work)

    def upsert_relationships_batch(
        self,
        rel_type: str,
        rels_data: List[Dict[str, Any]],
    ) -> int:
        """
        Transactional bulk upsert of relationships using UNWIND.
        """
        if not rels_data:
            return 0

        valid_type = validate_relationship_type(rel_type)
        clean_batch = []
        now_iso = datetime.utcnow().isoformat() + "Z"

        for item in rels_data:
            src = item.get("source_id")
            tgt = item.get("target_id")
            if not src or not tgt:
                raise ValueError(f"Every relationship must have 'source_id' and 'target_id'. Item: {item}")
            clean_item = {
                "source_id": validate_urn_identifier(str(src)),
                "target_id": validate_urn_identifier(str(tgt)),
                "props": item.get("properties", {}),
            }
            clean_batch.append(clean_item)

        cypher = f"""
        UNWIND $batch AS item
        MATCH (s {{id: item.source_id}})
        MATCH (t {{id: item.target_id}})
        MERGE (s)-[r:{valid_type}]->(t)
        ON CREATE SET r = item.props, r.created_at = $now_iso
        ON MATCH SET r += item.props, r.updated_at = $now_iso
        RETURN count(r) AS total
        """

        def _work(tx):
            res = tx.run(cypher, batch=clean_batch, now_iso=now_iso)
            record = res.single()
            return record["total"] if record else 0

        return self.db.execute_write(_work)


# Module-level convenience store instance
graph_store = GraphStore()
