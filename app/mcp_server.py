"""MCP server exposing project-scoped Graph-RAG tools to coding agents.

Run with ``python -m app.mcp_server``.  The default transport is stdio, which
is supported by Claude Code, Cursor, VS Code/Copilot, Codex, and other MCP
clients.  This module deliberately exposes graph operations rather than raw
Cypher or Neo4j credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.graph.answering.generator import GraphAnswerGenerator
from app.graph.ingestion.projects import ProjectIngestionService
from app.graph.retrieval.retriever import GraphRetriever
from app.graph.retrieval.specs import QueryIntent, RetrievalSpec, TraversalDirection
from app.graph.store import graph_store


mcp = FastMCP(
    name="Graph-RAG Project Intelligence",
    instructions=(
        "Use these tools to inspect the selected project's repositories, code graph, "
        "and uploaded documents. Results are grounded in Neo4j graph evidence. "
        "Create or ingest only when the user explicitly requests it."
    ),
)


def _service() -> ProjectIngestionService:
    return ProjectIngestionService(store=graph_store)


def _error(message: str) -> dict[str, Any]:
    """Return predictable tool errors without leaking driver or credential details."""
    return {"ok": False, "error": message}


def _project_exists(project_id: str) -> bool:
    return bool(graph_store.get_node(project_id, "Project"))


@mcp.tool(
    name="list_projects",
    description="List Graph-RAG projects available to the coding agent.",
)
def list_projects() -> dict[str, Any]:
    try:
        projects = _service().list_projects()
        return {
            "ok": True,
            "projects": [
                {
                    "project_id": project["id"],
                    "name": project.get("name", project["id"]),
                    "description": project.get("description", ""),
                }
                for project in projects
            ],
        }
    except Exception:
        return _error("The knowledge graph is unavailable.")


@mcp.tool(
    name="get_project_overview",
    description=(
        "Return the repository structure and bounded evidence graph for a project. "
        "Use this first when orienting to an unfamiliar project."
    ),
)
def get_project_overview(project_id: str) -> dict[str, Any]:
    try:
        if not _project_exists(project_id):
            return _error("Project does not exist. Call list_projects to obtain a valid project_id.")
        result = GraphRetriever().retrieve(
            RetrievalSpec(
                intent=QueryIntent.REPOSITORY_LOOKUP,
                direction=TraversalDirection.OUTGOING,
                relationship_types=["CONTAINS", "EXPOSES"],
                entity_types=["Repository"],
                max_depth=2,
                max_nodes=150,
                scope_project_urn=project_id,
            )
        )
        return {
            "ok": True,
            "project_id": project_id,
            "evidence": result.model_dump(mode="json"),
        }
    except Exception:
        return _error("Unable to retrieve the project overview from the knowledge graph.")


@mcp.tool(
    name="query_project_graph",
    description=(
        "Answer a natural-language question using only graph evidence from one project. "
        "Returns the grounded answer together with cited nodes and relationships."
    ),
)
def query_project_graph(project_id: str, question: str) -> dict[str, Any]:
    if not question or not question.strip():
        return _error("question must not be empty.")
    try:
        if not _project_exists(project_id):
            return _error("Project does not exist. Call list_projects to obtain a valid project_id.")
        answer = GraphAnswerGenerator().generate_answer(
            question=question.strip(),
            scope_project_urn=project_id,
        )
        return {"ok": True, **answer.model_dump(mode="json")}
    except Exception:
        return _error("Unable to answer the question from the knowledge graph.")


@mcp.tool(
    name="get_repository_evidence",
    description=(
        "Return the bounded graph evidence for one repository within a project, including files, "
        "symbols, APIs, and their relationships."
    ),
)
def get_repository_evidence(project_id: str, repository_id: str) -> dict[str, Any]:
    try:
        if not _project_exists(project_id):
            return _error("Project does not exist. Call list_projects to obtain a valid project_id.")
        result = GraphRetriever().retrieve(
            RetrievalSpec(
                intent=QueryIntent.REPOSITORY_LOOKUP,
                entity_mentions=[repository_id],
                entity_types=["Repository"],
                direction=TraversalDirection.OUTGOING,
                relationship_types=["CONTAINS", "EXPOSES"],
                max_depth=2,
                max_nodes=150,
                scope_project_urn=project_id,
            )
        )
        return {"ok": True, "project_id": project_id, "evidence": result.model_dump(mode="json")}
    except Exception:
        return _error("Unable to retrieve repository evidence from the knowledge graph.")


@mcp.tool(
    name="create_project",
    description="Create a new empty Graph-RAG project. Use only when the user explicitly requests a new project.",
)
def create_project(name: str, description: str = "") -> dict[str, Any]:
    try:
        project = _service().create_project(name=name, description=description)
        return {"ok": True, "project": project}
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        return _error("Unable to create project.")


@mcp.tool(
    name="ingest_repository",
    description=(
        "Ingest a Git URL or a local repository inside the MCP server's workspace into a project. "
        "This writes code entities and relationships to Neo4j; use only on explicit user request."
    ),
)
def ingest_repository(project_id: str, source: str, branch: str | None = None) -> dict[str, Any]:
    try:
        summary = _service().ingest_repository(project_id=project_id, source=source, branch=branch)
        return {"ok": True, "summary": summary}
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        return _error("Repository ingestion failed.")


@mcp.tool(
    name="ingest_project_documents",
    description=(
        "Ingest supported documents from a directory inside the MCP server's workspace. "
        "Supported formats: PDF, DOCX, PPTX, HTML, HTM, Markdown, and TXT. This writes to Neo4j."
    ),
)
def ingest_project_documents(project_id: str, directory: str) -> dict[str, Any]:
    try:
        documents_dir = Path(directory).expanduser().resolve()
        workspace = Path.cwd().resolve()
        if not documents_dir.is_dir():
            return _error("directory must be an existing directory.")
        if not documents_dir.is_relative_to(workspace):
            return _error("Document directories must be inside the MCP server workspace.")
        summary = _service().ingest_documents(project_id=project_id, documents_dir=documents_dir)
        return {"ok": True, "summary": summary}
    except ValueError as exc:
        return _error(str(exc))
    except Exception:
        return _error("Document ingestion failed.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
