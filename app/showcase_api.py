"""Read-only FastAPI showcase for the Graph-RAG knowledge graph."""

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph.answering.generator import GraphAnswerGenerator
from app.graph.connection import db_manager
from app.graph.retrieval.retriever import GraphRetriever
from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.store import graph_store


WEB_DIR = Path(__file__).parent / "web"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def get_answer_generator() -> GraphAnswerGenerator:
    return GraphAnswerGenerator()


def get_graph_retriever() -> GraphRetriever:
    return GraphRetriever()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Graph-RAG Showcase",
        version="0.1.0",
        description="Read-only, graph-only repository intelligence demo.",
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def landing_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return db_manager.health_check()

    @app.get("/api/repositories")
    def repositories() -> list[dict[str, str | None]]:
        try:
            nodes = graph_store.list_nodes_by_label("Repository", limit=100)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The knowledge graph is unavailable.") from exc
        return [
            {
                "id": node["id"],
                "name": node.get("name", node["id"]),
                "project_id": node.get("project_id"),
                "provider": node["id"].split(":")[2] if len(node["id"].split(":")) > 2 else None,
            }
            for node in nodes
        ]

    @app.get("/api/graph")
    def repository_graph(
        repository_id: str = Query(min_length=1, max_length=500),
        retriever: GraphRetriever = Depends(get_graph_retriever),
    ) -> dict[str, Any]:
        """Return a bounded evidence subgraph for one repository."""
        try:
            result = retriever.retrieve(
                RetrievalSpec(
                    intent=QueryIntent.REPOSITORY_LOOKUP,
                    entity_mentions=[repository_id],
                    entity_types=["Repository"],
                    relationship_types=["CONTAINS"],
                    direction=TraversalDirection.OUTGOING,
                    max_depth=2,
                    max_nodes=100,
                )
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Unable to retrieve repository evidence.") from exc
        return result.model_dump(mode="json")

    @app.post("/api/query")
    def ask_question(
        request: QuestionRequest,
        generator: GraphAnswerGenerator = Depends(get_answer_generator),
    ) -> dict[str, Any]:
        try:
            return generator.generate_answer(request.question).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Unable to answer the question from the knowledge graph.") from exc

    return app


app = create_app()
