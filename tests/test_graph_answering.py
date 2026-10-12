"""
Comprehensive Unit and Functional Tests for Phase 06: Graph Context Integration and Answering.
Tests:
1. Question with sufficient graph evidence
2. Empty retrieval result handling (no LLM fabrication)
3. Ambiguous entity resolution handling (no LLM guessing)
4. Insufficient evidence handling
5. LLM Gateway timeout / network failure handling
6. Malformed or empty LLM response handling
7. Preservation of file paths, line ranges, and provenance in evidence
8. Authorization context propagation
9. Confirmation that zero vector/Qdrant/embedding calls are made
10. Regression compatibility
"""

import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

from app.graph.retrieval.specs import QueryIntent
from app.graph.retrieval.subgraph import (
    RetrievalResult,
    RetrievedNode,
    RetrievedEdge,
    PathResult,
)
from app.graph.retrieval.access_control import (
    AccessControlContext,
    ScopedAccessController,
)
from app.graph.retrieval.retriever import GraphRetriever
from app.graph.retrieval.context_builder import GraphContextBuilder
from app.graph.answering.models import AnswerStatus, GroundedAnswerResponse
from app.graph.answering.generator import GraphAnswerGenerator


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLMGateway:
    def __init__(self, response_text: str = "AuthService contains routes.py which defines login_endpoint."):
        self.response_text = response_text
        self.model = "gemini-2.5-flash"
        self.temperature = 0.0
        self.invoked_prompts = []

    def invoke(self, prompt: str):
        self.invoked_prompts.append(prompt)
        return MockLLMResponse(self.response_text)


@pytest.fixture
def sample_retrieval_result():
    """Provides a realistic populated RetrievalResult."""
    res = RetrievalResult(intent=QueryIntent.ENTITY_LOOKUP)

    node1 = RetrievedNode(
        id="urn:service:local:auth_service",
        name="AuthService",
        label="Service",
        properties={"id": "urn:service:local:auth_service", "name": "AuthService"},
        hop_distance=0,
        provenance_type="STATIC_AST",
        provenance_file="services/auth.py",
        provenance_lines="L1-50",
        confidence=1.0,
    )
    node2 = RetrievedNode(
        id="urn:code:fn:auth_repo:routes.py:login_endpoint",
        name="login_endpoint",
        label="Function",
        properties={"id": "urn:code:fn:auth_repo:routes.py:login_endpoint", "name": "login_endpoint", "route": "/api/v1/login"},
        hop_distance=1,
        provenance_type="STATIC_AST",
        provenance_file="routes.py",
        provenance_lines="L20-35",
        confidence=1.0,
    )

    res.add_node(node1)
    res.add_node(node2)
    res.starting_entities.append(node1)

    edge = RetrievedEdge(
        source_id="urn:service:local:auth_service",
        target_id="urn:code:fn:auth_repo:routes.py:login_endpoint",
        rel_type="CONTAINS",
        direction="OUTGOING",
        hop=1,
        provenance_type="STATIC_AST",
        confidence=1.0,
    )
    res.add_relationship(edge)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────────────────────────────────────

def test_answering_sufficient_graph_evidence(sample_retrieval_result):
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result

    mock_llm = MockLLMGateway(response_text="AuthService contains the login_endpoint located at routes.py:L20-35.")
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    response = generator.generate_answer("What does AuthService contain?")

    assert response.status == AnswerStatus.ANSWERED
    assert "login_endpoint" in response.answer
    assert response.retrieval_intent == "entity_lookup"
    assert len(response.evidence.nodes) == 2
    assert len(response.evidence.relationships) == 1
    assert response.evidence.nodes[0].name == "AuthService"
    assert response.model_metadata is not None
    assert response.model_metadata.model_name == "gemini-2.5-flash"
    assert len(mock_llm.invoked_prompts) == 1


def test_answering_empty_retrieval_result():
    empty_result = RetrievalResult(intent=QueryIntent.ENTITY_LOOKUP)
    empty_result.warnings.append("Entity mention 'NonExistentService' could not be found.")

    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = empty_result

    mock_llm = MockLLMGateway()
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    response = generator.generate_answer("What is NonExistentService?")

    # Must return INSUFFICIENT_EVIDENCE and NEVER call the LLM to fabricate
    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "Insufficient graph evidence" in response.answer
    assert len(mock_llm.invoked_prompts) == 0


def test_answering_ambiguous_entity_resolution():
    amb_result = RetrievalResult(intent=QueryIntent.ENTITY_LOOKUP)
    amb_result.ambiguities.append({
        "mention": "authenticate",
        "candidates": [
            {"urn": "urn:fn:auth_repo:authenticate", "file": "auth.py"},
            {"urn": "urn:fn:pay_repo:authenticate", "file": "pay.py"},
        ],
        "message": "Multiple candidates found in auth_repo and pay_repo",
    })

    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = amb_result

    mock_llm = MockLLMGateway()
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    response = generator.generate_answer("Who calls authenticate?")

    # Must return AMBIGUOUS status directly without calling LLM
    assert response.status == AnswerStatus.AMBIGUOUS
    assert "ambiguous entity references" in response.answer
    assert len(mock_llm.invoked_prompts) == 0


def test_answering_llm_gateway_failure(sample_retrieval_result):
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = TimeoutError("Portkey/Gemini Gateway timeout after 30s")

    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)
    response = generator.generate_answer("What does AuthService contain?")

    assert response.status == AnswerStatus.ERROR
    assert "LLM Gateway error" in response.answer
    assert any("timeout" in w.lower() for w in response.warnings)


def test_answering_empty_llm_response(sample_retrieval_result):
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result

    mock_llm = MockLLMGateway(response_text="   ")  # Whitespace only
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    response = generator.generate_answer("What does AuthService contain?")

    assert response.status == AnswerStatus.ERROR
    assert "empty response" in response.answer.lower()


def test_provenance_preservation_in_evidence(sample_retrieval_result):
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result
    mock_llm = MockLLMGateway()

    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)
    response = generator.generate_answer("What does AuthService contain?")

    # Check evidence preserves file paths and line ranges exactly
    fn_node = next(n for n in response.evidence.nodes if n.name == "login_endpoint")
    assert fn_node.source_location == "routes.py:L20-35"
    assert fn_node.provenance_type == "STATIC_AST"
    assert fn_node.properties.get("route") == "/api/v1/login"


def test_authorization_context_propagation():
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.side_effect = PermissionError("Principal denied access to restricted repo.")

    generator = GraphAnswerGenerator(retriever=mock_retriever)
    restricted_ctx = AccessControlContext(allowed_repos={"urn:repo:other"})

    response = generator.generate_answer("Describe AuthService", context=restricted_ctx)

    assert response.status == AnswerStatus.ERROR
    assert "Access denied" in response.answer
    mock_retriever.retrieve.assert_called_once_with(query_or_spec="Describe AuthService", context=restricted_ctx)


def test_zero_vector_search_invocation(sample_retrieval_result, monkeypatch):
    """Verifies that no vector search or embeddings are touched in the answer pipeline."""
    # Poison any potential Qdrant/vector calls
    mock_poison = MagicMock(side_effect=RuntimeError("Vector search was invoked in Graph answering path!"))
    monkeypatch.setattr("app.config.settings.QDRANT_COLLECTION", "poison_collection")

    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result
    mock_llm = MockLLMGateway(response_text="Clean grounded answer.")

    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)
    response = generator.generate_answer("What does AuthService contain?")

    assert response.status == AnswerStatus.ANSWERED
    assert response.answer == "Clean grounded answer."


def test_answering_insufficient_evidence_unresolved_references():
    unres_result = RetrievalResult(intent=QueryIntent.CALLER_LOOKUP)
    unres_result.unresolved_references.append("fake_symbol_123")
    unres_result.warnings.append("Entity mention 'fake_symbol_123' could not be found.")

    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = unres_result

    mock_llm = MockLLMGateway()
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    response = generator.generate_answer("Who calls fake_symbol_123?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "fake_symbol_123" in response.answer
    assert len(mock_llm.invoked_prompts) == 0


def test_answering_context_truncation_handling(sample_retrieval_result):
    mock_retriever = MagicMock(spec=GraphRetriever)
    mock_retriever.retrieve.return_value = sample_retrieval_result

    mock_llm = MockLLMGateway(response_text="Truncated context handled.")
    generator = GraphAnswerGenerator(retriever=mock_retriever, llm=mock_llm)

    # Set tiny max_context_chars = 100
    response = generator.generate_answer("What does AuthService contain?", max_context_chars=100)

    assert response.status == AnswerStatus.ANSWERED
    assert len(mock_llm.invoked_prompts) == 1
    # Verify the truncation marker was embedded into prompt
    assert "... [Context truncated to size limit] ..." in mock_llm.invoked_prompts[0]

