"""
Graph Context Answering Package.
Exports the grounded answer generation service, response data models,
and grounding prompts.
"""

from app.graph.answering.models import (
    AnswerStatus,
    GroundedAnswerResponse,
    EvidenceSummary,
    EvidenceNodeSummary,
    EvidenceRelationshipSummary,
    ModelMetadata,
)
from app.graph.answering.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    build_answering_prompt,
)
from app.graph.answering.generator import GraphAnswerGenerator

__all__ = [
    "AnswerStatus",
    "GroundedAnswerResponse",
    "EvidenceSummary",
    "EvidenceNodeSummary",
    "EvidenceRelationshipSummary",
    "ModelMetadata",
    "GROUNDED_SYSTEM_PROMPT",
    "build_answering_prompt",
    "GraphAnswerGenerator",
]
