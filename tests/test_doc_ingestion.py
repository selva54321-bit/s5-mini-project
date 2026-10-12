"""
Comprehensive Test Suite for Document, Architecture, and Policy Ingestion (Phase 04).
Tests multi-format document parsing, policy/security-rule extraction, architecture components,
cross-domain entity linking, ambiguity detection, provenance, and pipeline execution.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from app.graph.schema import NodeLabel, RelationshipType, ProvenanceType
from app.graph.ingestion.docs.identifiers import (
    build_doc_urn,
    build_policy_urn,
    build_security_rule_urn,
    build_arch_component_urn,
)
from app.graph.ingestion.docs.parser import (
    discover_documents,
    parse_document,
    compute_file_hash,
)
from app.graph.ingestion.docs.policy_extractor import PolicyExtractor
from app.graph.ingestion.docs.architecture_extractor import ArchitectureExtractor
from app.graph.ingestion.docs.linker import CrossDomainLinker, AmbiguousMatch
from app.graph.ingestion.docs.pipeline import ingest_documents


# ── 1. Identifier Generation Tests ────────────────────────────────────────────

def test_doc_and_policy_identifiers():
    doc_urn = build_doc_urn("urn:project:acme:cloud", "architecture/overview.md")
    assert doc_urn == "urn:doc:acme:cloud:architecture/overview.md"

    pol_urn = build_policy_urn("urn:company:acme", "COMP-K8S-GOV-01")
    assert pol_urn == "urn:policy:acme:COMP-K8S-GOV-01"

    rule_urn = build_security_rule_urn(pol_urn, "job-must-have-resources")
    assert rule_urn == "urn:secrule:acme:COMP-K8S-GOV-01:job-must-have-resources"

    arch_urn = build_arch_component_urn("urn:project:acme:cloud", "nginx-frontend")
    assert arch_urn == "urn:arch:acme:cloud:nginx-frontend"


# ── 2. Document Discovery & Parsing Tests ─────────────────────────────────────

def test_document_parser_markdown():
    sample_md = """# Architecture Overview
This is the enterprise system overview.

## Data Ingestion
Ingests documents and source code into the unified graph.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(sample_md)
        f_path = f.name

    try:
        doc = parse_document(abs_path=f_path, rel_path="docs/arch.md", project_urn="urn:project:acme:test")
        assert doc.success is True
        assert doc.title == "Architecture Overview"
        assert len(doc.sections) >= 2
        assert "Data Ingestion" in [s.heading for s in doc.sections]
        assert doc.file_size_bytes > 0
        assert len(doc.content_hash) == 64
    finally:
        os.remove(f_path)


def test_document_parser_html():
    sample_html = """<!DOCTYPE html>
<html>
<head><title>System Spec</title></head>
<body>
<h1>Kubernetes Cluster Spec</h1>
<p>Cluster runs in private VPC.</p>
<h2>Security Context</h2>
<p>Containers must run as non-root user.</p>
</body>
</html>"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(sample_html)
        f_path = f.name

    try:
        doc = parse_document(abs_path=f_path, rel_path="specs/k8s.html", project_urn="urn:project:acme:test")
        assert doc.success is True
        assert "Kubernetes Cluster Spec" in doc.full_text
        assert "non-root" in doc.full_text
    finally:
        os.remove(f_path)


def test_document_parser_handles_empty_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("")
        f_path = f.name

    try:
        doc = parse_document(abs_path=f_path, rel_path="empty.txt", project_urn="urn:project:acme:test")
        assert doc.success is False
        assert "Empty file" in doc.error_message
    finally:
        os.remove(f_path)


# ── 3. Policy & Security Rule Extraction Tests ────────────────────────────────

SAMPLE_COMPLIANCE_DOC = """# Compliance and Governance
1. Audit Logging
Enhanced logging for compliance:
Job Template labels: audit-log: "enabled", compliance-required: "true"

2. Policy Enforcement with OPA Gatekeeper
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: jobrequiresresources
msg := "All job containers must define resource requests and limits for compliance"

Security Context:
runAsNonRoot: true
allowPrivilegeEscalation: false
capabilities:
  drop: ALL

3. Data Retention Policies
retention_policies = {
    "application_logs": timedelta(days=90),
    "audit_logs": timedelta(days=2555),
    "user_data": timedelta(days=1825)
}
"""

def test_policy_extractor_extracts_rules_with_provenance():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE_COMPLIANCE_DOC)
        f_path = f.name

    try:
        # Parse document
        doc = parse_document(abs_path=f_path, rel_path="policies/compliance.md", project_urn="urn:project:acme:test")
        extractor = PolicyExtractor(company_urn="urn:company:acme")
        policies = extractor.extract_from_document(doc)

        pol_codes = [p.code for p in policies]
        assert "COMP-K8S-GOV-01" in pol_codes
        assert "COMP-DATA-RET-01" in pol_codes

        k8s_pol = next(p for p in policies if p.code == "COMP-K8S-GOV-01")
        rule_names = [r.name for r in k8s_pol.rules]
        assert "job-must-have-resources" in rule_names
        assert "enforce-non-root-execution" in rule_names
        assert "audit-log-enabled" in rule_names

        # Verify evidence snippet and severity
        res_rule = next(r for r in k8s_pol.rules if r.name == "job-must-have-resources")
        assert res_rule.severity == "HIGH"
        assert "resource requests and limits" in res_rule.rule_text
        assert res_rule.confidence == 1.0

        ret_pol = next(p for p in policies if p.code == "COMP-DATA-RET-01")
        assert len(ret_pol.rules) == 3  # application_logs, audit_logs, user_data

    finally:
        os.remove(f_path)


# ── 4. Architecture Knowledge Extraction Tests ────────────────────────────────

SAMPLE_ARCH_DOC = """# System Architecture Overview
Frontend Ingress Tier:
Labels:
  app: nginx
  tier: frontend
Description: nginx frontend proxy routing requests to backend api.

Event-Driven Jobs with KEDA:
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: queue-processor
spec:
  jobTargetRef:
    template: processor
"""

def test_architecture_extractor_extracts_components():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE_ARCH_DOC)
        f_path = f.name

    try:
        doc = parse_document(abs_path=f_path, rel_path="architecture.md", project_urn="urn:project:acme:test")
        extractor = ArchitectureExtractor(project_urn="urn:project:acme:test")
        components = extractor.extract_from_document(doc)

        comp_names = [c.name for c in components]
        assert "Nginx Frontend Tier" in comp_names
        assert "KEDA Queue Processor" in comp_names

        nginx = next(c for c in components if c.name == "Nginx Frontend Tier")
        assert nginx.component_type == "ingress_proxy"
        assert "backend-api-service" in nginx.dependencies
        assert nginx.confidence >= 0.95

    finally:
        os.remove(f_path)


# ── 5. Cross-Domain Entity Linking & Ambiguity Detection Tests ────────────────

def test_cross_domain_linker_resolves_and_detects_ambiguity():
    doc = parse_document(
        abs_path="dummy.md",
        rel_path="docs/sec.md",
        project_urn="urn:project:acme:test"
    )
    doc.success = True
    doc.full_text = SAMPLE_COMPLIANCE_DOC

    extractor = PolicyExtractor(company_urn="urn:company:acme")
    policies = extractor.extract_from_document(doc)

    # Known code entities:
    # 1. "compliance-report-job" has exactly 1 candidate -> resolved
    # 2. "job" has multiple candidates -> flagged as ambiguous!
    known_entities = {
        "compliance-report-job": ["urn:service:acme:compliance-report-job"],
        "job": ["urn:service:acme:batch-job-1", "urn:service:acme:batch-job-2"],
    }

    linker = CrossDomainLinker(known_code_entities=known_entities)
    result = linker.link_all(
        docs=[doc],
        policies=policies,
        components=[],
        project_urn="urn:project:acme:test",
    )

    rel_types = [r.rel_type for r in result.relationships]
    assert RelationshipType.DOCUMENTED_BY.value in rel_types
    assert RelationshipType.CONTAINS.value in rel_types
    assert RelationshipType.APPLIES_TO.value in rel_types
    assert RelationshipType.GOVERNED_BY.value in rel_types

    # Verify verified link
    verified_link = next(
        r for r in result.relationships
        if r.rel_type == RelationshipType.APPLIES_TO.value and "compliance-report-job" in r.target_id
    )
    assert verified_link.source_id.startswith("urn:secrule:")
    assert verified_link.target_id == "urn:service:acme:compliance-report-job"

    # Verify ambiguous match is recorded and NOT guessed
    assert len(result.ambiguous_matches) > 0
    ambig = result.ambiguous_matches[0]
    assert len(ambig.candidate_urns) == 2
    assert "Multiple code services" in ambig.reason


# ── 6. End-to-End Pipeline Dry-Run & Idempotency Tests ────────────────────────

def test_pipeline_dry_run_on_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic docs
        doc1 = os.path.join(tmpdir, "architecture.md")
        doc2 = os.path.join(tmpdir, "policy.txt")

        with open(doc1, "w") as f:
            f.write(SAMPLE_ARCH_DOC)
        with open(doc2, "w") as f:
            f.write(SAMPLE_COMPLIANCE_DOC)

        summary = ingest_documents(
            docs_path=tmpdir,
            project_urn="urn:project:acme:cloud",
            company_urn="urn:company:acme",
            write_to_db=False,
        )

        assert summary.documents_discovered == 2
        assert summary.documents_parsed == 2
        assert summary.documents_skipped == 0
        assert len(summary.errors) == 0

        # Check extracted entities
        assert summary.nodes_created_or_updated[NodeLabel.DOCUMENT.value] == 2
        assert summary.nodes_created_or_updated[NodeLabel.POLICY.value] >= 2
        assert summary.nodes_created_or_updated[NodeLabel.SECURITY_RULE.value] >= 4
        assert summary.nodes_created_or_updated[NodeLabel.ARCHITECTURE_COMPONENT.value] >= 1


def test_pipeline_database_write_with_mock_store():
    mock_store = MagicMock()
    mock_store.upsert_nodes_batch.return_value = 4
    mock_store.upsert_relationships_batch.return_value = 6

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "sec.md"), "w") as f:
            f.write(SAMPLE_COMPLIANCE_DOC)

        summary = ingest_documents(
            docs_path=tmpdir,
            project_urn="urn:project:acme:cloud",
            company_urn="urn:company:acme",
            write_to_db=True,
            store=mock_store,
        )

        assert summary.documents_parsed == 1
        assert len(summary.errors) == 0
        assert mock_store.upsert_nodes_batch.called
        assert mock_store.upsert_relationships_batch.called
