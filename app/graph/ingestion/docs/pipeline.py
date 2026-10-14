"""
Document, Architecture, and Policy Ingestion Pipeline (Task 5 & Task 6).
Coordinates multi-format document discovery, parsing, policy & architecture extraction,
cross-domain linking with code entities, and idempotent Neo4j persistence.
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
from app.graph.ingestion.docs.identifiers import (
    build_doc_urn,
    build_policy_urn,
    build_arch_component_urn,
)
from app.graph.ingestion.docs.parser import (
    discover_documents,
    parse_document,
    ParsedDocument,
)
from app.graph.ingestion.docs.policy_extractor import PolicyExtractor, ExtractedPolicy
from app.graph.ingestion.docs.architecture_extractor import (
    ArchitectureExtractor,
    ExtractedArchComponent,
)
from app.graph.ingestion.docs.linker import CrossDomainLinker, LinkerResult

logger = logging.getLogger("graph.doc_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class DocIngestionSummary:
    project_urn: str
    documents_discovered: int = 0
    documents_parsed: int = 0
    documents_skipped: int = 0
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)
    nodes_created_or_updated: Dict[str, int] = field(default_factory=dict)
    relationships_created_or_updated: Dict[str, int] = field(default_factory=dict)
    ambiguous_entity_matches: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_urn": self.project_urn,
            "documents_discovered": self.documents_discovered,
            "documents_parsed": self.documents_parsed,
            "documents_skipped": self.documents_skipped,
            "parse_failures_count": len(self.parse_failures),
            "nodes_created_or_updated": self.nodes_created_or_updated,
            "relationships_created_or_updated": self.relationships_created_or_updated,
            "ambiguous_entity_matches": self.ambiguous_entity_matches,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def ingest_documents(
    docs_path: str,
    project_urn: str = "urn:project:local:default",
    company_urn: str = "urn:company:local:default",
    known_code_entities: Optional[Dict[str, List[str]]] = None,
    write_to_db: bool = True,
    store: Optional[GraphStore] = None,
) -> DocIngestionSummary:
    """
    Main programmatic entrypoint for scanning and ingesting organizational documents,
    architecture components, and security policies into the knowledge graph.
    """
    start_time = time.time()
    abs_docs_path = os.path.abspath(docs_path)
    summary = DocIngestionSummary(project_urn=project_urn)

    logger.info(f"Starting document ingestion at '{abs_docs_path}' for project: {project_urn}")

    # 1. Document Discovery
    doc_paths = discover_documents(abs_docs_path)
    summary.documents_discovered = len(doc_paths)
    logger.info(f"Discovered {len(doc_paths)} candidate documents.")

    # 2. Document Parsing (Task 1)
    parsed_docs: List[ParsedDocument] = []
    seen_hashes = set()

    for abs_file in doc_paths:
        rel_path = os.path.relpath(abs_file, abs_docs_path).replace("\\", "/")
        p_doc = parse_document(abs_path=abs_file, rel_path=rel_path, project_urn=project_urn)

        if p_doc.success:
            # Check for duplicate document content
            if p_doc.content_hash in seen_hashes:
                summary.documents_skipped += 1
                summary.warnings.append(f"Skipping duplicate document content: {rel_path}")
                continue
            seen_hashes.add(p_doc.content_hash)

            summary.documents_parsed += 1
            parsed_docs.append(p_doc)
        else:
            summary.documents_skipped += 1
            summary.parse_failures.append({
                "rel_path": rel_path,
                "error": p_doc.error_message,
            })
            summary.warnings.append(f"Failed to parse {rel_path}: {p_doc.error_message}")

    # 3. Policy & Security Rule Extraction (Task 3)
    policy_extractor = PolicyExtractor(company_urn=company_urn)
    all_policies: List[ExtractedPolicy] = []
    for doc in parsed_docs:
        extracted_pols = policy_extractor.extract_from_document(doc)
        all_policies.extend(extracted_pols)

    # 4. Architecture Knowledge Extraction (Task 2)
    arch_extractor = ArchitectureExtractor(project_urn=project_urn)
    all_components: List[ExtractedArchComponent] = []
    for doc in parsed_docs:
        extracted_comps = arch_extractor.extract_from_document(doc)
        all_components.extend(extracted_comps)

    # 5. Cross-Domain Entity Linking (Task 4)
    linker = CrossDomainLinker(known_code_entities=known_code_entities)
    link_result: LinkerResult = linker.link_all(
        docs=parsed_docs,
        policies=all_policies,
        components=all_components,
        project_urn=project_urn,
    )
    summary.ambiguous_entity_matches = len(link_result.ambiguous_matches)

    # 6. Prepare Node Payloads
    node_batches: Dict[str, List[Dict[str, Any]]] = {
        NodeLabel.DOCUMENT.value: [],
        NodeLabel.ARCHITECTURE_COMPONENT.value: [],
        NodeLabel.POLICY.value: [],
        NodeLabel.SECURITY_RULE.value: [],
    }

    # Document Nodes
    for doc in parsed_docs:
        node_batches[NodeLabel.DOCUMENT.value].append({
            "id": doc.urn,
            "title": doc.title,
            "source_type": doc.source_type,
            "file_path": doc.rel_path,
            "total_pages": doc.total_pages,
            "content_hash": doc.content_hash,
            "project_id": project_urn,
        })

    # Architecture Component Nodes
    for comp in all_components:
        node_batches[NodeLabel.ARCHITECTURE_COMPONENT.value].append({
            "id": comp.urn,
            "name": comp.name,
            "component_type": comp.component_type,
            "description": comp.description,
            "project_id": project_urn,
        })

    # Policy & Security Rule Nodes
    for pol in all_policies:
        node_batches[NodeLabel.POLICY.value].append({
            "id": pol.urn,
            "code": pol.code,
            "title": pol.title,
            "category": pol.category,
            "company_id": company_urn,
            "project_id": project_urn,
        })
        for rule in pol.rules:
            node_batches[NodeLabel.SECURITY_RULE.value].append({
                "id": rule.urn,
                "name": rule.name,
                "policy_id": pol.urn,
                "rule_text": rule.rule_text,
                "severity": rule.severity,
                "project_id": project_urn,
            })

    # Group Relationships by Type
    rel_batches: Dict[str, List[Dict[str, Any]]] = {}
    for r in link_result.relationships:
        rel_batches.setdefault(r.rel_type, []).append({
            "source_id": r.source_id,
            "target_id": r.target_id,
            "properties": {**r.properties, **r.provenance},
        })

    # 7. Write to Neo4j (Task 5)
    target_store = store or graph_store

    if write_to_db:
        logger.info("Writing document knowledge and cross-domain links to Neo4j...")
        try:
            for label, batch in node_batches.items():
                if batch:
                    count = target_store.upsert_nodes_batch(label=label, nodes_data=batch)
                    summary.nodes_created_or_updated[label] = count

            for rel_type, batch in rel_batches.items():
                if batch:
                    count = target_store.upsert_relationships_batch(rel_type=rel_type, rels_data=batch)
                    summary.relationships_created_or_updated[rel_type] = count

            logger.info("Document knowledge ingestion committed successfully.")
        except Exception as e:
            err = f"Failed to persist document knowledge to Neo4j: {e}"
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
        f"Document ingestion completed in {summary.duration_seconds:.2f}s: "
        f"{sum(summary.nodes_created_or_updated.values())} nodes, "
        f"{sum(summary.relationships_created_or_updated.values())} relationships."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation, architecture, and policies into Neo4j.")
    parser.add_argument("docs_path", help="Local directory path of documents to ingest.")
    parser.add_argument("--project", default="urn:project:local:default", help="Project URN.")
    parser.add_argument("--company", default="urn:company:local:default", help="Company URN.")
    parser.add_argument("--no-db", action="store_true", help="Perform extraction without writing to Neo4j.")
    args = parser.parse_args()

    summary = ingest_documents(
        docs_path=args.docs_path,
        project_urn=args.project,
        company_urn=args.company,
        write_to_db=not args.no_db,
    )

    print("\n" + "=" * 60)
    print("           DOCUMENT INGESTION SUMMARY")
    print("=" * 60)
    print(f"Project URN:            {summary.project_urn}")
    print(f"Documents Discovered:   {summary.documents_discovered}")
    print(f"Documents Parsed:       {summary.documents_parsed}")
    print(f"Documents Skipped:      {summary.documents_skipped}")
    print(f"Ambiguous Matches:      {summary.ambiguous_entity_matches}")
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
