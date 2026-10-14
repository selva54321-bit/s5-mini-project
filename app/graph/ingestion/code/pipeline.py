"""
Repository Code Ingestion Pipeline (Task 6 & Task 7).
Coordinates repository scanning, AST parsing, bounded symbol resolution,
and idempotent transactional ingestion into Neo4j via GraphStore.
"""

import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.graph.schema import NodeLabel, RelationshipType, ProvenanceType
from app.graph.store import GraphStore, graph_store
from app.graph.ingestion.code.identifiers import (
    build_repo_urn,
    build_package_urn,
)
from app.graph.ingestion.code.scanner import scan_repository, DiscoveredFile
from app.graph.ingestion.code.parser import parse_python_file, ParseResult
from app.graph.ingestion.code.resolver import SymbolResolver, ResolvedRelationship
from app.graph.ingestion.code.git_source import is_git_url, clone_repository, cleanup_cloned_repo, GitSourceError

logger = logging.getLogger("graph.code_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class IngestionSummary:
    repo_urn: str
    files_discovered: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)
    nodes_created_or_updated: Dict[str, int] = field(default_factory=dict)
    relationships_created_or_updated: Dict[str, int] = field(default_factory=dict)
    unresolved_references: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_urn": self.repo_urn,
            "files_discovered": self.files_discovered,
            "files_parsed": self.files_parsed,
            "files_skipped": self.files_skipped,
            "parse_failures_count": len(self.parse_failures),
            "nodes_created_or_updated": self.nodes_created_or_updated,
            "relationships_created_or_updated": self.relationships_created_or_updated,
            "unresolved_references": self.unresolved_references,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def ingest_repository(
    repo_path: str,
    repo_name: Optional[str] = None,
    org: str = "local",
    provider: str = "git",
    project_id: Optional[str] = None,
    write_to_db: bool = True,
    store: Optional[GraphStore] = None,
    branch: Optional[str] = None,
    cleanup_after: bool = True,
) -> IngestionSummary:
    """
    Main programmatic entrypoint for scanning and ingesting a code repository into the graph.

    Accepts either a local directory path or a remote git URL (HTTPS/SSH).
    When a git URL is provided, the repository is cloned to a temporary directory,
    ingested, and optionally cleaned up after completion.
    """
    start_time = time.time()
    cloned_path = None  # Track if we cloned, for cleanup

    # Handle git URLs: clone to local temp directory
    if is_git_url(repo_path):
        logger.info(f"Detected git URL: {repo_path}")
        try:
            cloned_path, auto_name, auto_org, auto_provider = clone_repository(
                git_url=repo_path, branch=branch,
            )
            abs_repo_path = cloned_path
            name = repo_name or auto_name
            org = auto_org if org == "local" else org
            provider = auto_provider
        except GitSourceError as e:
            summary = IngestionSummary(repo_urn=f"urn:repo:git:{org}:{repo_name or 'unknown'}")
            summary.errors.append(f"Git clone failed: {e}")
            summary.duration_seconds = time.time() - start_time
            return summary
    else:
        abs_repo_path = os.path.abspath(repo_path)
        name = repo_name or os.path.basename(abs_repo_path)

    repo_urn = build_repo_urn(name, org=org, provider=provider)
    summary = IngestionSummary(repo_urn=repo_urn)
    logger.info(f"Starting code ingestion for repository: '{name}' (URN: {repo_urn}) at '{abs_repo_path}'")

    # 1. Repository Discovery (Task 1)
    try:
        discovered_files = scan_repository(abs_repo_path)
        summary.files_discovered = len(discovered_files)
        logger.info(f"Discovered {len(discovered_files)} supported source files.")
    except Exception as e:
        err = f"Failed repository discovery on '{abs_repo_path}': {e}"
        logger.error(err)
        summary.errors.append(err)
        summary.duration_seconds = time.time() - start_time
        return summary

    # 2. Syntactic Language Parsing (Task 2 & Task 3)
    parse_results: List[ParseResult] = []
    for d_file in discovered_files:
        p_res = parse_python_file(
            abs_path=d_file.abs_path,
            rel_path=d_file.rel_path,
            repo_urn=repo_urn,
        )
        if p_res.success:
            summary.files_parsed += 1
            parse_results.append(p_res)
        else:
            summary.files_skipped += 1
            summary.parse_failures.append({
                "rel_path": d_file.rel_path,
                "error": p_res.error_message,
            })
            summary.warnings.append(f"Skipped {d_file.rel_path}: {p_res.error_message}")

    # 3. Bounded Symbol Resolution (Task 4 & Task 5)
    resolver = SymbolResolver(repo_urn=repo_urn, parse_results=parse_results)
    resolved_rels, unresolved_refs = resolver.resolve_all()
    summary.unresolved_references = len(unresolved_refs)
    logger.info(
        f"Resolution complete: {len(resolved_rels)} relationships resolved, "
        f"{len(unresolved_refs)} unresolved symbol references."
    )

    # 4. Prepare Entity Nodes
    node_batches: Dict[str, List[Dict[str, Any]]] = {
        NodeLabel.REPOSITORY.value: [
            {
                "id": repo_urn,
                "name": name,
                "path": abs_repo_path,
                "project_id": project_id or f"urn:project:{org}:default",
            }
        ],
        NodeLabel.FILE.value: [],
        NodeLabel.CLASS.value: [],
        NodeLabel.FUNCTION.value: [],
        NodeLabel.API.value: [],
        NodeLabel.TEST.value: [],
        NodeLabel.PACKAGE.value: [],
    }

    external_packages_seen = set()
    api_handler_urns = {
        api.handler_func_urn
        for parse_result in parse_results
        for api in parse_result.apis
    }

    for pr in parse_results:
        # File Node
        node_batches[NodeLabel.FILE.value].append({
            "id": pr.file_urn,
            "name": os.path.basename(pr.rel_path),
            "path": pr.rel_path,
            "loc": pr.loc,
            "repo_id": repo_urn,
            "extension": ".py",
            "workflow_nodes": pr.workflow_nodes,
            "workflow_node_count": len(pr.workflow_nodes),
        })

        # Class Nodes
        for c in pr.classes:
            node_batches[NodeLabel.CLASS.value].append({
                "id": c.urn,
                "name": c.name,
                "qualified_name": c.qualified_name,
                "file_id": pr.file_urn,
                "repo_id": repo_urn,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "docstring": c.docstring or "",
            })

        # Function Nodes
        for f in pr.functions:
            node_batches[NodeLabel.FUNCTION.value].append({
                "id": f.urn,
                "name": f.name,
                "qualified_name": f.qualified_name,
                "file_id": pr.file_urn,
                "repo_id": repo_urn,
                "is_method": f.is_method,
                "is_async": f.is_async,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "parameters": f.parameters,
                "docstring": f.docstring or "",
                "source_excerpt": f.source_excerpt,
                "is_api_handler": f.urn in api_handler_urns,
            })

        # API Nodes
        for api in pr.apis:
            node_batches[NodeLabel.API.value].append({
                "id": api.urn,
                "route": api.route,
                "method": api.http_method,
                "framework": api.framework,
                "repo_id": repo_urn,
                "handler_id": api.handler_func_urn,
            })

        # Test Nodes
        for t in pr.tests:
            node_batches[NodeLabel.TEST.value].append({
                "id": t.urn,
                "name": t.name,
                "file_id": pr.file_urn,
                "repo_id": repo_urn,
                "test_type": "unit",
                "line_start": t.line_start,
            })

        # Packages
        for imp in pr.imports:
            pkg_name = imp.module.split(".")[0]
            if pkg_name and pkg_name not in resolver.ctx.module_to_file and pkg_name not in external_packages_seen:
                external_packages_seen.add(pkg_name)
                pkg_urn = build_package_urn(pkg_name)
                node_batches[NodeLabel.PACKAGE.value].append({
                    "id": pkg_urn,
                    "name": pkg_name,
                    "ecosystem": "pypi",
                })

    # Group Relationships by Type
    rel_batches: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved_rels:
        rel_batches.setdefault(r.rel_type, []).append({
            "source_id": r.source_id,
            "target_id": r.target_id,
            "properties": {**r.properties, **r.provenance},
        })

    # 5. Write to Neo4j (Task 6)
    target_store = store or graph_store

    if write_to_db:
        logger.info("Writing extracted entities and relationships to Neo4j...")
        try:
            # Upsert Nodes by Label
            for label, batch in node_batches.items():
                if batch:
                    count = target_store.upsert_nodes_batch(label=label, nodes_data=batch)
                    summary.nodes_created_or_updated[label] = count

            # Upsert Relationships by Type
            for rel_type, batch in rel_batches.items():
                if batch:
                    count = target_store.upsert_relationships_batch(rel_type=rel_type, rels_data=batch)
                    summary.relationships_created_or_updated[rel_type] = count

            logger.info("Database ingestion successfully committed.")
        except Exception as e:
            err = f"Failed to persist graph data to Neo4j: {e}"
            logger.error(err)
            summary.errors.append(err)
    else:
        logger.info("Dry-run mode: skipping database write.")
        for label, batch in node_batches.items():
            summary.nodes_created_or_updated[label] = len(batch)
        for rel_type, batch in rel_batches.items():
            summary.relationships_created_or_updated[rel_type] = len(batch)

    summary.duration_seconds = time.time() - start_time
    logger.info(
        f"Ingestion finished in {summary.duration_seconds:.2f}s: "
        f"{sum(summary.nodes_created_or_updated.values())} nodes, "
        f"{sum(summary.relationships_created_or_updated.values())} relationships."
    )

    # Cleanup cloned repository if applicable
    if cloned_path and cleanup_after:
        try:
            cleanup_cloned_repo(cloned_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup cloned repo at {cloned_path}: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest a code repository into the Neo4j Knowledge Graph.")
    parser.add_argument("repo_path", help="Local directory path or remote git URL (HTTPS/SSH) of repository to ingest.")
    parser.add_argument("--repo-name", help="Repository name override.")
    parser.add_argument("--org", default="local", help="Organization slug.")
    parser.add_argument("--branch", default=None, help="Git branch or tag to checkout (only for git URLs).")
    parser.add_argument("--no-db", action="store_true", help="Perform extraction without writing to Neo4j.")
    parser.add_argument("--keep-clone", action="store_true", help="Do not delete cloned repo after ingestion.")
    args = parser.parse_args()

    summary = ingest_repository(
        repo_path=args.repo_path,
        repo_name=args.repo_name,
        org=args.org,
        write_to_db=not args.no_db,
        branch=args.branch,
        cleanup_after=not args.keep_clone,
    )

    print("\n" + "=" * 60)
    print("           REPOSITORY INGESTION SUMMARY")
    print("=" * 60)
    print(f"Repository URN:         {summary.repo_urn}")
    print(f"Files Discovered:       {summary.files_discovered}")
    print(f"Files Parsed:           {summary.files_parsed}")
    print(f"Files Skipped:          {summary.files_skipped}")
    print(f"Unresolved References:  {summary.unresolved_references}")
    print(f"Duration:               {summary.duration_seconds:.2f}s")
    print("\n--- Entities Extracted ---")
    for k, v in summary.nodes_created_or_updated.items():
        print(f"  {k:22}: {v}")
    print("\n--- Relationships Extracted ---")
    for k, v in summary.relationships_created_or_updated.items():
        print(f"  {k:22}: {v}")
    if summary.errors:
        print("\n--- Errors ---")
        for e in summary.errors:
            print(f"  ❌ {e}")
    if summary.warnings:
        print("\n--- Warnings ---")
        for w in summary.warnings:
            print(f"  ⚠️ {w}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
