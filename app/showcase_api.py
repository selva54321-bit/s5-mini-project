"""FastAPI showcase for project-scoped Graph-RAG ingestion and retrieval."""

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph.answering.generator import GraphAnswerGenerator
from app.graph.connection import db_manager
from app.graph.retrieval.retriever import GraphRetriever
from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.store import graph_store
from app.graph.ingestion.projects import ProjectIngestionService


WEB_DIR = Path(__file__).parent / "web"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    project_id: str = Field(min_length=1, max_length=500)


class ProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class RepositoryIngestionRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=1000)
    branch: str | None = Field(default=None, max_length=200)


def get_answer_generator() -> GraphAnswerGenerator:
    return GraphAnswerGenerator()


def get_graph_retriever() -> GraphRetriever:
    return GraphRetriever()


def get_project_ingestion_service() -> ProjectIngestionService:
    return ProjectIngestionService(store=graph_store)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Graph-RAG Showcase",
        version="0.1.0",
        description="Project-scoped, graph-only repository intelligence demo.",
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def landing_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return db_manager.health_check()

    @app.get("/api/repositories")
    def repositories(project_id: str | None = None) -> list[dict[str, str | None]]:
        try:
            nodes = graph_store.list_nodes_by_label("Repository", limit=100)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The knowledge graph is unavailable.") from exc
        repos = [
            {
                "id": node["id"],
                "name": node.get("name", node["id"]),
                "project_id": node.get("project_id"),
                "provider": node["id"].split(":")[2] if len(node["id"].split(":")) > 2 else None,
            }
            for node in nodes
        ]
        return [repo for repo in repos if not project_id or repo["project_id"] == project_id]

    @app.get("/api/projects")
    def projects(service: ProjectIngestionService = Depends(get_project_ingestion_service)) -> list[dict[str, Any]]:
        try:
            return service.list_projects()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The knowledge graph is unavailable.") from exc

    @app.post("/api/projects", status_code=201)
    def create_project(request: ProjectRequest, service: ProjectIngestionService = Depends(get_project_ingestion_service)) -> dict[str, Any]:
        try:
            return service.create_project(request.name, request.description)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Unable to create project.") from exc

    @app.post("/api/ingest/repository")
    def ingest_project_repository(
        request: RepositoryIngestionRequest,
        service: ProjectIngestionService = Depends(get_project_ingestion_service),
    ) -> dict[str, Any]:
        try:
            return service.ingest_repository(request.project_id, request.source, request.branch)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Repository ingestion failed.") from exc

    @app.post("/api/ingest/documents")
    async def ingest_project_documents(
        project_id: str = Form(...),
        files: list[UploadFile] = File(...),
        service: ProjectIngestionService = Depends(get_project_ingestion_service),
    ) -> dict[str, Any]:
        try:
            saved_dir: Path | None = None
            for upload in files:
                saved_dir = service.save_document(project_id, upload.filename or "", await upload.read())
            if saved_dir is None:
                raise ValueError("Upload at least one document.")
            return service.ingest_documents(project_id, saved_dir)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Document ingestion failed.") from exc

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
            return generator.generate_answer(
                request.question,
                scope_project_urn=request.project_id,
            ).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Unable to answer the question from the knowledge graph.") from exc

    return app


app = create_app()
