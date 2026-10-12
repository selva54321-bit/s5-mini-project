"""
Idempotent Neo4j Schema Initializer for Phase 01 / Phase 02.
Applies approved uniqueness constraints, lookup indexes, and full-text indexes.
Never drops databases, existing indexes, or user data.
"""

import logging
from typing import Dict, Any, List

from app.graph.connection import db_manager
from app.graph.schema import (
    APPROVED_UNIQUENESS_CONSTRAINTS,
    APPROVED_LOOKUP_INDEXES,
    APPROVED_FULLTEXT_INDEXES,
)

logger = logging.getLogger("graph.schema_init")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def initialize_schema() -> Dict[str, Any]:
    """
    Applies all approved constraints and indexes idempotently using 'IF NOT EXISTS'.
    Returns a comprehensive execution report.
    """
    logger.info("Starting idempotent Neo4j schema initialization...")

    results = {
        "constraints_applied": [],
        "lookup_indexes_applied": [],
        "fulltext_indexes_applied": [],
        "errors": [],
    }

    with db_manager.session() as session:
        # 1. Uniqueness Constraints
        for c_stmt in APPROVED_UNIQUENESS_CONSTRAINTS:
            try:
                session.run(c_stmt)
                results["constraints_applied"].append(c_stmt)
                logger.debug(f"Applied constraint: {c_stmt}")
            except Exception as e:
                err_msg = f"Failed constraint '{c_stmt}': {e}"
                logger.error(err_msg)
                results["errors"].append(err_msg)

        # 2. Lookup B-Tree Indexes
        for idx_stmt in APPROVED_LOOKUP_INDEXES:
            try:
                session.run(idx_stmt)
                results["lookup_indexes_applied"].append(idx_stmt)
                logger.debug(f"Applied index: {idx_stmt}")
            except Exception as e:
                err_msg = f"Failed index '{idx_stmt}': {e}"
                logger.error(err_msg)
                results["errors"].append(err_msg)

        # 3. Full-Text Indexes
        for ft_stmt in APPROVED_FULLTEXT_INDEXES:
            try:
                session.run(ft_stmt)
                results["fulltext_indexes_applied"].append(ft_stmt)
                logger.debug(f"Applied full-text index: {ft_stmt}")
            except Exception as e:
                err_msg = f"Failed full-text index '{ft_stmt}': {e}"
                logger.error(err_msg)
                results["errors"].append(err_msg)

    total_applied = (
        len(results["constraints_applied"])
        + len(results["lookup_indexes_applied"])
        + len(results["fulltext_indexes_applied"])
    )
    logger.info(
        f"Schema initialization completed: {total_applied} statements applied, "
        f"{len(results['errors'])} errors."
    )
    return results


def verify_active_schema() -> Dict[str, List[Dict[str, Any]]]:
    """
    Queries active constraints and indexes from the Neo4j catalog.
    """
    with db_manager.session() as session:
        constraints_result = session.run("SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties")
        constraints = [dict(record) for record in constraints_result]

        indexes_result = session.run("SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state")
        indexes = [dict(record) for record in indexes_result]

    return {
        "constraints": constraints,
        "indexes": indexes,
    }


if __name__ == "__main__":
    try:
        report = initialize_schema()
        print(f"Schema Initialization Summary: {len(report['constraints_applied'])} constraints, {len(report['lookup_indexes_applied'])} indexes applied.")
        if report["errors"]:
            print(f"Encountered {len(report['errors'])} errors:")
            for err in report["errors"]:
                print(f"  - {err}")
    except Exception as exc:
        print(f"Schema initialization failed: {exc}")
