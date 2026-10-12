"""
Structured Response and Evidence Models for Graph Context Answering.
Defines typed response objects, evidence summaries, provenance tracking,
and answer status enums adhering to project validation standards.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class EvidenceNodeSummary(BaseModel):
    id: str
    name: str
    label: str
    source_location: Optional[str] = None
    provenance_type: Optional[str] = None
    confidence: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class EvidenceRelationshipSummary(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    direction: str = "OUTGOING"
    hop: int = 1
    provenance_type: Optional[str] = None
    confidence: float = 1.0


class EvidenceSummary(BaseModel):
    nodes: List[EvidenceNodeSummary] = Field(default_factory=list)
    relationships: List[EvidenceRelationshipSummary] = Field(default_factory=list)
    paths: List[str] = Field(default_factory=list)
    provenance_counts: Dict[str, int] = Field(default_factory=dict)


class ModelMetadata(BaseModel):
    model_name: str
    temperature: float = 0.0
    provider: str = "google_genai"
    cache_status: Optional[str] = None
    latency_ms: Optional[float] = None


class GroundedAnswerResponse(BaseModel):
    """
    Typed, structured response produced by GraphAnswerGenerator.
    Provides verifiable traceability back to graph traversal evidence.
    """
    question: str
    answer: str
    status: AnswerStatus
    retrieval_intent: str
    entities_resolved: List[str] = Field(default_factory=list)
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)
    warnings: List[str] = Field(default_factory=list)
    model_metadata: Optional[ModelMetadata] = None
