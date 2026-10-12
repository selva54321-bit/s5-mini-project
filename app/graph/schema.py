"""
Approved Schema Definitions & Validation Rules for Phase 01 / Phase 02.
Defines entity labels, relationship types, provenance tiers, and schema objects.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class NodeLabel(str, Enum):
    # Organizational
    COMPANY = "Company"
    PROJECT = "Project"
    REPOSITORY = "Repository"
    TEAM = "Team"
    DEVELOPER = "Developer"

    # Software Structure
    SERVICE = "Service"
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    API = "API"
    DATABASE = "Database"
    TABLE = "Table"
    PACKAGE = "Package"
    TEST = "Test"
    ARCHITECTURE_COMPONENT = "ArchitectureComponent"

    # Organizational Knowledge & Governance
    DOCUMENT = "Document"
    POLICY = "Policy"
    SECURITY_RULE = "SecurityRule"
    CODING_STANDARD = "CodingStandard"
    COMMIT = "Commit"


class RelationshipType(str, Enum):
    BELONGS_TO = "BELONGS_TO"
    CONTAINS = "CONTAINS"
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    EXPOSES = "EXPOSES"
    READS = "READS"
    WRITES = "WRITES"
    TESTED_BY = "TESTED_BY"
    DOCUMENTED_BY = "DOCUMENTED_BY"
    GOVERNED_BY = "GOVERNED_BY"
    REQUIRES = "REQUIRES"
    APPLIES_TO = "APPLIES_TO"
    MODIFIED_BY = "MODIFIED_BY"
    CHANGED_WITH = "CHANGED_WITH"


class ProvenanceType(str, Enum):
    STATIC_AST = "STATIC_AST"
    GIT_HISTORY = "GIT_HISTORY"
    DOC_EXPLICIT = "DOC_EXPLICIT"
    LLM_EXTRACTED = "LLM_EXTRACTED"
    HEURISTIC = "HEURISTIC"
    MANUAL = "MANUAL"


# Whitelist sets for O(1) schema validation
ALLOWED_NODE_LABELS = frozenset(e.value for e in NodeLabel)
ALLOWED_RELATIONSHIPS = frozenset(e.value for e in RelationshipType)
ALLOWED_PROVENANCE_TYPES = frozenset(e.value for e in ProvenanceType)


def validate_node_label(label: str) -> str:
    """
    Validates a node label against the approved Phase 01 schema.
    Prevents Cypher injection when interpolating validated labels into queries.
    """
    if not label or not isinstance(label, str):
        raise ValueError("Node label must be a non-empty string.")
    cleaned = label.strip()
    if cleaned not in ALLOWED_NODE_LABELS:
        raise ValueError(
            f"Invalid node label '{cleaned}'. Must be one of: {sorted(ALLOWED_NODE_LABELS)}"
        )
    return cleaned


def validate_relationship_type(rel_type: str) -> str:
    """
    Validates a relationship type against the approved Phase 01 schema.
    Prevents Cypher injection when interpolating validated types into queries.
    """
    if not rel_type or not isinstance(rel_type, str):
        raise ValueError("Relationship type must be a non-empty string.")
    cleaned = rel_type.strip().upper()
    if cleaned not in ALLOWED_RELATIONSHIPS:
        raise ValueError(
            f"Invalid relationship type '{cleaned}'. Must be one of: {sorted(ALLOWED_RELATIONSHIPS)}"
        )
    return cleaned


def validate_urn_identifier(entity_id: str) -> str:
    """
    Validates that an entity identifier follows the approved URN pattern.
    Format: urn:<domain>:<hierarchy_tokens>:<scoped_identifier>
    """
    if not entity_id or not isinstance(entity_id, str):
        raise ValueError("Entity identifier must be a non-empty string.")
    cleaned = entity_id.strip()
    if not cleaned.startswith("urn:"):
        raise ValueError(
            f"Entity ID '{cleaned}' does not match required URN format ('urn:<category>:...')"
        )
    return cleaned


class ProvenanceMetadata(BaseModel):
    provenance_type: ProvenanceType = ProvenanceType.STATIC_AST
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_repo_id: Optional[str] = None
    source_project_id: Optional[str] = None
    source_file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    commit_hash: Optional[str] = None
    doc_page_number: Optional[int] = None
    evidence_snippet: Optional[str] = None
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extractor_version: str = "1.0.0"


class NodeEntity(BaseModel):
    id: str
    label: NodeLabel
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[ProvenanceMetadata] = None

    @field_validator("id")
    @classmethod
    def check_id(cls, v: str) -> str:
        return validate_urn_identifier(v)


class RelationshipEntity(BaseModel):
    rel_type: RelationshipType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[ProvenanceMetadata] = None

    @field_validator("source_id", "target_id")
    @classmethod
    def check_endpoints(cls, v: str) -> str:
        return validate_urn_identifier(v)


# --- APPROVED SCHEMA STATEMENTS (Idempotent Cypher DDL) ---

APPROVED_UNIQUENESS_CONSTRAINTS = [
    # Schema-enforced uniqueness on id for each approved node label
    f"CREATE CONSTRAINT c_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
    for label in sorted(ALLOWED_NODE_LABELS)
]

APPROVED_LOOKUP_INDEXES = [
    # Fast repository & project scoping
    "CREATE INDEX idx_file_repo IF NOT EXISTS FOR (n:File) ON (n.repo_id)",
    "CREATE INDEX idx_file_path IF NOT EXISTS FOR (n:File) ON (n.path)",
    # Fast symbol resolution by qualified name
    "CREATE INDEX idx_func_qname IF NOT EXISTS FOR (n:Function) ON (n.qualified_name)",
    "CREATE INDEX idx_class_qname IF NOT EXISTS FOR (n:Class) ON (n.qualified_name)",
    # Fast route matching for APIs
    "CREATE INDEX idx_api_route IF NOT EXISTS FOR (n:API) ON (n.route, n.method)",
    # Policy code lookup
    "CREATE INDEX idx_policy_code IF NOT EXISTS FOR (n:Policy) ON (n.code)",
    "CREATE INDEX idx_rule_name IF NOT EXISTS FOR (n:SecurityRule) ON (n.name)",
]

APPROVED_FULLTEXT_INDEXES = [
    # Full-text search for agent symbol resolution
    "CREATE FULLTEXT INDEX fts_symbols IF NOT EXISTS FOR (n:Function|Class|API|Service) ON EACH [n.name, n.qualified_name, n.docstring]",
    # Full-text search across documentation and policy rules
    "CREATE FULLTEXT INDEX fts_knowledge IF NOT EXISTS FOR (n:Document|Policy|SecurityRule|ArchitectureComponent) ON EACH [n.name, n.title, n.rule_text, n.description]",
]
