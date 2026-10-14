from fastapi.testclient import TestClient

from app.showcase_api import create_app, get_answer_generator, get_graph_retriever
from app.graph.answering.models import AnswerStatus, GroundedAnswerResponse
from app.graph.retrieval.subgraph import RetrievalResult
from app.graph.retrieval.specs import QueryIntent


def test_showcase_exposes_read_only_api_surface():
    app = create_app()
    paths = {route.path for route in app.routes}
    assert {"/", "/api/health", "/api/repositories", "/api/graph", "/api/query"} <= paths


def test_question_endpoint_returns_grounded_response():
    class FakeGenerator:
        def generate_answer(self, question):
            return GroundedAnswerResponse(
                question=question,
                answer="Grounded answer",
                status=AnswerStatus.ANSWERED,
                retrieval_intent="entity_lookup",
            )

    app = create_app()
    app.dependency_overrides[get_answer_generator] = lambda: FakeGenerator()
    response = TestClient(app).post(
        "/api/query",
        json={"question": "What is eqrev?", "project_id": "urn:project:local:demo"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer"


def test_graph_endpoint_returns_serialized_subgraph():
    class FakeRetriever:
        def retrieve(self, spec):
            return RetrievalResult(intent=QueryIntent.REPOSITORY_LOOKUP)

    app = create_app()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeRetriever()
    response = TestClient(app).get("/api/graph", params={"repository_id": "urn:repo:git:local:app"})

    assert response.status_code == 200
    assert response.json()["intent"] == "repository_lookup"
