"""
Document, Architecture, and Policy Ingestion Package (Phase 04)
"""
from app.graph.ingestion.docs.identifiers import (
    build_doc_urn,
    build_policy_urn,
    build_security_rule_urn,
    build_coding_standard_urn,
    build_arch_component_urn,
)
from app.graph.ingestion.docs.parser import discover_documents, parse_document, ParsedDocument
from app.graph.ingestion.docs.policy_extractor import PolicyExtractor, ExtractedPolicy, ExtractedSecurityRule
from app.graph.ingestion.docs.architecture_extractor import ArchitectureExtractor, ExtractedArchComponent
from app.graph.ingestion.docs.linker import CrossDomainLinker, LinkerResult, AmbiguousMatch
from app.graph.ingestion.docs.pipeline import ingest_documents, DocIngestionSummary
