"""
Deterministic Graph Traversal Engine.
Executes reviewed, parameterized Cypher templates for direct neighbors, bounded
multi-hop traversal, shortest paths, impact analysis, caller/callee exploration,
and test/policy connections. Enforces cycle prevention, depth limits, and timeout.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel

from app.graph.store import GraphStore, graph_store
from app.graph.schema import (
    validate_node_label,
    validate_relationship_type,
    validate_urn_identifier,
    ALLOWED_NODE_LABELS,
    ALLOWED_RELATIONSHIPS,
)
from app.graph.retrieval.specs import TraversalDirection
from app.graph.retrieval.subgraph import (
    RetrievedNode,
    RetrievedEdge,
    PathResult,
)
from app.graph.retrieval.access_control import (
    AccessControlContext,
    AccessController,
    default_access_controller,
)

logger = logging.getLogger("graph.retrieval.traversal")


def _format_rel_filter(rel_types: Optional[List[str]]) -> str:
    """Validates and formats relationship type filter for Cypher pattern."""
    if not rel_types:
        return ""
    valid_types = [validate_relationship_type(r) for r in rel_types if r]
    if not valid_types:
        return ""
    return ":" + "|".join(valid_types)


def _extract_node_from_record(node_dict: Dict[str, Any], labels: List[str], hop: int = 0) -> RetrievedNode:
    """Extracts a RetrievedNode with provenance details from Neo4j record dictionary."""
    node_id = str(node_dict.get("id", ""))
    primary_label = labels[0] if labels else "Unknown"
    name = str(node_dict.get("name") or node_id)

    # Provenance fields
    p_type = node_dict.get("provenance_provenance_type") or node_dict.get("provenance_type")
    p_file = node_dict.get("provenance_source_file_path") or node_dict.get("source_file_path") or node_dict.get("file_path")
    line_start = node_dict.get("provenance_line_start") or node_dict.get("line_start")
    line_end = node_dict.get("provenance_line_end") or node_dict.get("line_end")
    lines_str = f"L{line_start}-{line_end}" if line_start and line_end else (f"L{line_start}" if line_start else None)
    confidence = float(node_dict.get("provenance_confidence") or 1.0)

    clean_props = {k: v for k, v in node_dict.items() if not k.startswith("provenance_")}

    return RetrievedNode(
        id=node_id,
        label=primary_label,
        name=name,
        properties=clean_props,
        hop_distance=hop,
        provenance_type=p_type,
        provenance_file=p_file,
        provenance_lines=lines_str,
        confidence=confidence,
    )


class GraphTraversalEngine:
    """
    Executes reviewed Cypher traversal queries safely and parses paths into Subgraph structures.
    """

    def __init__(
        self,
        store: Optional[GraphStore] = None,
        access_controller: Optional[AccessController] = None,
    ):
        self.store = store or graph_store
        self.access_controller = access_controller or default_access_controller

    def get_direct_neighbors(
        self,
        source_id: str,
        direction: TraversalDirection = TraversalDirection.BOTH,
        rel_types: Optional[List[str]] = None,
        limit: int = 50,
        context: Optional[AccessControlContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves direct 1-hop neighbors of an entity.
        """
        valid_src = validate_urn_identifier(source_id)
        rel_clause = _format_rel_filter(rel_types)
        ctx = context or AccessControlContext()

        if direction == TraversalDirection.OUTGOING:
            pattern = f"(s {{id: $source_id}})-[r{rel_clause}]->(t)"
        elif direction == TraversalDirection.INCOMING:
            pattern = f"(s {{id: $source_id}})<-[r{rel_clause}]-(t)"
        else:
            pattern = f"(s {{id: $source_id}})-[r{rel_clause}]-(t)"

        cypher = f"""
        MATCH {pattern}
        RETURN s, labels(s) AS s_labels,
               r, type(r) AS rel_type, properties(r) AS rel_props,
               startNode(r).id AS rel_source_id, endNode(r).id AS rel_target_id,
               t, labels(t) AS t_labels
        LIMIT $limit
        """

        def _work(tx):
            res = tx.run(cypher, source_id=valid_src, limit=min(limit, 200))
            rows = []
            for record in res:
                s_dict = dict(record["s"])
                t_dict = dict(record["t"])
                rel_type = record["rel_type"]
                if not self.access_controller.can_traverse_relationship(s_dict, rel_type, t_dict, ctx):
                    continue
                rows.append({
                    "source": _extract_node_from_record(s_dict, record["s_labels"], hop=0),
                    "target": _extract_node_from_record(t_dict, record["t_labels"], hop=1),
                    "rel_type": rel_type,
                    "rel_source_id": record["rel_source_id"],
                    "rel_target_id": record["rel_target_id"],
                    "rel_props": dict(record["rel_props"]),
                })
            return rows

        return self.store.db.execute_read(_work)

    def find_bounded_paths(
        self,
        source_id: str,
        direction: TraversalDirection = TraversalDirection.OUTGOING,
        rel_types: Optional[List[str]] = None,
        max_depth: int = 2,
        max_paths: int = 20,
        context: Optional[AccessControlContext] = None,
    ) -> List[PathResult]:
        """
        Explores bounded multi-hop paths from source_id. Enforces cycle prevention.
        """
        valid_src = validate_urn_identifier(source_id)
        bounded_depth = max(1, min(max_depth, 5))  # Hard ceiling 5
        rel_clause = _format_rel_filter(rel_types)
        ctx = context or AccessControlContext()

        if direction == TraversalDirection.OUTGOING:
            pattern = f"(s {{id: $source_id}})-[r{rel_clause}*1..{bounded_depth}]->(t)"
        elif direction == TraversalDirection.INCOMING:
            pattern = f"(s {{id: $source_id}})<-[r{rel_clause}*1..{bounded_depth}]-(t)"
        else:
            pattern = f"(s {{id: $source_id}})-[r{rel_clause}*1..{bounded_depth}]-(t)"

        cypher = f"""
        MATCH p = {pattern}
        WHERE none(n IN nodes(p)[1..] WHERE n.id = s.id)
        RETURN p, length(p) AS hop_length
        ORDER BY hop_length ASC
        LIMIT $max_paths
        """

        def _work(tx):
            res = tx.run(cypher, source_id=valid_src, max_paths=min(max_paths, 100))
            path_results = []
            for idx, record in enumerate(res):
                path_obj = record["p"]
                hop_len = record["hop_length"]
                nodes_in_path = []
                edges_in_path = []

                # Parse path nodes and relationships
                raw_nodes = path_obj.nodes
                raw_rels = path_obj.relationships

                # Access control check for all nodes in path
                allowed = True
                for n in raw_nodes:
                    if not self.access_controller.can_access_node(dict(n), ctx):
                        allowed = False
                        break
                if not allowed:
                    continue

                for h_idx, n in enumerate(raw_nodes):
                    node_ret = _extract_node_from_record(dict(n), list(n.labels), hop=h_idx)
                    nodes_in_path.append(node_ret)

                for r in raw_rels:
                    edge_ret = RetrievedEdge(
                        source_id=r.start_node["id"],
                        target_id=r.end_node["id"],
                        rel_type=r.type,
                        properties=dict(r),
                        direction="OUTGOING" if r.start_node["id"] == valid_src else "INCOMING",
                        hop=hop_len,
                        provenance_type=r.get("provenance_type"),
                        confidence=float(r.get("confidence") or 1.0),
                    )
                    edges_in_path.append(edge_ret)

                # Format human readable summary
                summary_parts = []
                for step_idx in range(len(nodes_in_path)):
                    summary_parts.append(f"({nodes_in_path[step_idx].name})")
                    if step_idx < len(edges_in_path):
                        summary_parts.append(f"-[:{edges_in_path[step_idx].rel_type}]->")

                path_results.append(
                    PathResult(
                        path_id=f"path_{idx + 1}",
                        hop_length=hop_len,
                        nodes=nodes_in_path,
                        relationships=edges_in_path,
                        summary="".join(summary_parts),
                    )
                )
            return path_results

        return self.store.db.execute_read(_work)

    def find_shortest_paths_between(
        self,
        source_id: str,
        target_id: str,
        rel_types: Optional[List[str]] = None,
        max_depth: int = 3,
        max_paths: int = 10,
        context: Optional[AccessControlContext] = None,
    ) -> List[PathResult]:
        """
        Finds all shortest paths between two resolved entities.
        """
        valid_src = validate_urn_identifier(source_id)
        valid_tgt = validate_urn_identifier(target_id)
        bounded_depth = max(1, min(max_depth, 5))
        rel_clause = _format_rel_filter(rel_types)
        ctx = context or AccessControlContext()

        cypher = f"""
        MATCH (s {{id: $source_id}}), (t {{id: $target_id}})
        MATCH p = allShortestPaths((s)-[r{rel_clause}*..{bounded_depth}]-(t))
        RETURN p, length(p) AS hop_length
        LIMIT $max_paths
        """

        def _work(tx):
            res = tx.run(
                cypher,
                source_id=valid_src,
                target_id=valid_tgt,
                max_paths=min(max_paths, 50),
            )
            paths = []
            for idx, record in enumerate(res):
                p_obj = record["p"]
                hop_len = record["hop_length"]

                raw_nodes = p_obj.nodes
                raw_rels = p_obj.relationships

                allowed = True
                for n in raw_nodes:
                    if not self.access_controller.can_access_node(dict(n), ctx):
                        allowed = False
                        break
                if not allowed:
                    continue

                nodes_in_path = [_extract_node_from_record(dict(n), list(n.labels), hop=h_idx) for h_idx, n in enumerate(raw_nodes)]
                edges_in_path = [
                    RetrievedEdge(
                        source_id=r.start_node["id"],
                        target_id=r.end_node["id"],
                        rel_type=r.type,
                        properties=dict(r),
                        hop=hop_len,
                    )
                    for r in raw_rels
                ]

                summary_parts = []
                for s_idx in range(len(nodes_in_path)):
                    summary_parts.append(f"({nodes_in_path[s_idx].name})")
                    if s_idx < len(edges_in_path):
                        summary_parts.append(f"-[:{edges_in_path[s_idx].rel_type}]->")

                paths.append(
                    PathResult(
                        path_id=f"path_{idx + 1}",
                        hop_length=hop_len,
                        nodes=nodes_in_path,
                        relationships=edges_in_path,
                        summary="".join(summary_parts),
                    )
                )
            return paths

        return self.store.db.execute_read(_work)

    def find_impact_chain(
        self,
        source_id: str,
        max_depth: int = 3,
        max_paths: int = 30,
        context: Optional[AccessControlContext] = None,
    ) -> List[PathResult]:
        """
        Direction-aware reverse traversal to identify components impacted by changes to source_id.
        Traverses incoming CALLS, IMPORTS, DEPENDS_ON edges.
        """
        valid_src = validate_urn_identifier(source_id)
        bounded_depth = max(1, min(max_depth, 5))
        ctx = context or AccessControlContext()

        # Dependent components point to source: (dependent)-[:CALLS|IMPORTS|DEPENDS_ON]->(source)
        cypher = f"""
        MATCH p = (dependent)-[r:CALLS|IMPORTS|DEPENDS_ON*1..{bounded_depth}]->(s {{id: $source_id}})
        WHERE none(n IN nodes(p)[..-1] WHERE n.id = s.id)
        RETURN p, length(p) AS hop_length
        ORDER BY hop_length ASC
        LIMIT $max_paths
        """

        def _work(tx):
            res = tx.run(cypher, source_id=valid_src, max_paths=min(max_paths, 100))
            paths = []
            for idx, record in enumerate(res):
                p_obj = record["p"]
                hop_len = record["hop_length"]

                raw_nodes = p_obj.nodes
                raw_rels = p_obj.relationships

                allowed = True
                for n in raw_nodes:
                    if not self.access_controller.can_access_node(dict(n), ctx):
                        allowed = False
                        break
                if not allowed:
                    continue

                nodes_in_path = [_extract_node_from_record(dict(n), list(n.labels), hop=h_idx) for h_idx, n in enumerate(raw_nodes)]
                edges_in_path = [
                    RetrievedEdge(
                        source_id=r.start_node["id"],
                        target_id=r.end_node["id"],
                        rel_type=r.type,
                        properties=dict(r),
                        hop=hop_len,
                    )
                    for r in raw_rels
                ]

                summary_parts = []
                for s_idx in range(len(nodes_in_path)):
                    summary_parts.append(f"({nodes_in_path[s_idx].name})")
                    if s_idx < len(edges_in_path):
                        summary_parts.append(f"-[:{edges_in_path[s_idx].rel_type}]->")

                paths.append(
                    PathResult(
                        path_id=f"impact_path_{idx + 1}",
                        hop_length=hop_len,
                        nodes=nodes_in_path,
                        relationships=edges_in_path,
                        summary="".join(summary_parts),
                    )
                )
            return paths

        return self.store.db.execute_read(_work)

    def find_tests_for(
        self,
        source_id: str,
        limit: int = 30,
        context: Optional[AccessControlContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves tests connected via TESTED_BY or calling the source node.
        """
        valid_src = validate_urn_identifier(source_id)
        ctx = context or AccessControlContext()

        cypher = """
        MATCH (s {id: $source_id})<-[r:TESTED_BY]-(test:Test)
        RETURN s, labels(s) AS s_labels, r, type(r) AS rel_type, properties(r) AS rel_props,
               test, labels(test) AS test_labels, test.id AS test_id, test.name AS test_name
        UNION
        MATCH (test:Test)-[r:CALLS]->(s {id: $source_id})
        RETURN s, labels(s) AS s_labels, r, type(r) AS rel_type, properties(r) AS rel_props,
               test, labels(test) AS test_labels, test.id AS test_id, test.name AS test_name
        LIMIT $limit
        """

        def _work(tx):
            res = tx.run(cypher, source_id=valid_src, limit=min(limit, 100))
            rows = []
            for record in res:
                test_dict = dict(record["test"])
                s_dict = dict(record["s"])
                if not self.access_controller.can_access_node(test_dict, ctx):
                    continue
                rows.append({
                    "test": _extract_node_from_record(test_dict, record["test_labels"], hop=1),
                    "target": _extract_node_from_record(s_dict, record["s_labels"], hop=0),
                    "rel_type": record["rel_type"],
                    "rel_props": dict(record["rel_props"]),
                })
            return rows

        return self.store.db.execute_read(_work)

    def find_policies_and_governance(
        self,
        source_id: str,
        limit: int = 30,
        context: Optional[AccessControlContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves policies, security rules, and coding standards governing or requiring the source node.
        """
        valid_src = validate_urn_identifier(source_id)
        ctx = context or AccessControlContext()

        cypher = """
        MATCH (s {id: $source_id})-[r:GOVERNED_BY|REQUIRES|APPLIES_TO|DOCUMENTED_BY]-(t)
        RETURN s, labels(s) AS s_labels, r, type(r) AS rel_type, properties(r) AS rel_props,
               startNode(r).id AS rel_source_id, endNode(r).id AS rel_target_id,
               t, labels(t) AS t_labels
        LIMIT $limit
        """

        def _work(tx):
            res = tx.run(cypher, source_id=valid_src, limit=min(limit, 100))
            rows = []
            for record in res:
                t_dict = dict(record["t"])
                s_dict = dict(record["s"])
                if not self.access_controller.can_access_node(t_dict, ctx):
                    continue
                rows.append({
                    "source": _extract_node_from_record(s_dict, record["s_labels"], hop=0),
                    "target": _extract_node_from_record(t_dict, record["t_labels"], hop=1),
                    "rel_type": record["rel_type"],
                    "rel_source_id": record["rel_source_id"],
                    "rel_target_id": record["rel_target_id"],
                    "rel_props": dict(record["rel_props"]),
                })
            return rows

        return self.store.db.execute_read(_work)
