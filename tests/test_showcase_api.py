from fastapi.testclient import TestClient

from app.showcase_api import (
    create_app,
    get_answer_generator,
    get_graph_retriever,
    get_project_ingestion_service,
)
from app.graph.answering.models import AnswerStatus, GroundedAnswerResponse
from app.graph.retrieval.subgraph import RetrievalResult
from app.graph.retrieval.specs import QueryIntent


def test_showcase_exposes_api_surface():
    app = create_app()
    paths = {route.path for route in app.routes}
    assert {"/", "/api/health", "/api/projects", "/api/projects/{project_id:path}", "/api/repositories", "/api/graph", "/api/query"} <= paths


def test_question_endpoint_returns_grounded_response():
    class FakeGenerator:
        def generate_answer(self, question, scope_project_urn=None):
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


def test_delete_project_calls_scoped_deletion_service():
    class FakeProjectService:
        def delete_project(self, project_id):
            return {"project_id": project_id, "deleted_nodes": 4, "uploads_deleted": True}

    app = create_app()
    app.dependency_overrides[get_project_ingestion_service] = lambda: FakeProjectService()
    response = TestClient(app).delete("/api/projects/urn:project:local:demo")

    assert response.status_code == 200
    assert response.json()["deleted_nodes"] == 4
