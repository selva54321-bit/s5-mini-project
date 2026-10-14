"""Project-scoped orchestration for repository and document ingestion."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.graph.ingestion.code.git_source import is_git_url
from app.graph.ingestion.code.pipeline import ingest_repository
from app.graph.ingestion.docs.parser import SUPPORTED_DOC_EXTENSIONS
from app.graph.ingestion.docs.pipeline import ingest_documents
from app.graph.schema import NodeLabel, RelationshipType, validate_urn_identifier
from app.graph.store import GraphStore, graph_store


UPLOAD_ROOT = Path("DATA") / "project_uploads"
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024


def build_project_urn(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Project name must include letters or numbers.")
    return f"urn:project:local:{slug}"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("A valid file name is required.")
    return name


@dataclass
class ProjectIngestionService:
    store: GraphStore = graph_store
    upload_root: Path = UPLOAD_ROOT

    def create_project(self, name: str, description: Optional[str] = None) -> dict[str, Any]:
        project_id = build_project_urn(name)
        return self.store.upsert_node(
            NodeLabel.PROJECT.value,
            {
                "id": project_id,
                "name": name.strip(),
                "description": (description or "").strip(),
                "project_id": project_id,
            },
        )

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_nodes_by_label(NodeLabel.PROJECT.value, limit=100)

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """Remove a project and all graph entities owned only by that project."""
        project_id = validate_urn_identifier(project_id)
        if not self.store.get_node(project_id, NodeLabel.PROJECT.value):
            raise ValueError("Project does not exist.")

        deleted_nodes = self.store.delete_project_scope(project_id)
        project_dir = (self.upload_root.resolve() / project_id.rsplit(":", 1)[-1]).resolve()
        upload_root = self.upload_root.resolve()
        uploads_deleted = False
        if project_dir.is_relative_to(upload_root) and project_dir.exists():
            shutil.rmtree(project_dir)
            uploads_deleted = True

        return {
            "project_id": project_id,
            "deleted_nodes": deleted_nodes,
            "uploads_deleted": uploads_deleted,
        }

    def ingest_repository(
        self,
        project_id: str,
        source: str,
        branch: Optional[str] = None,
    ) -> dict[str, Any]:
        project_id = validate_urn_identifier(project_id)
        if not self.store.get_node(project_id, NodeLabel.PROJECT.value):
            raise ValueError("Project does not exist.")

        source = source.strip()
        if not source:
            raise ValueError("Repository source is required.")
        if not is_git_url(source):
            local_path = Path(source).expanduser().resolve()
            workspace = Path.cwd().resolve()
            if not local_path.is_relative_to(workspace):
                raise ValueError("Local repositories must be inside the application workspace. Use a Git URL otherwise.")
            source = str(local_path)

        summary = ingest_repository(
            source,
            project_id=project_id,
            branch=branch,
            store=self.store,
        )
        if not summary.errors:
            self.store.upsert_relationship(
                RelationshipType.CONTAINS.value,
                project_id,
                summary.repo_urn,
                properties={"source_kind": "repository"},
            )
        return summary.to_dict()

    def save_document(self, project_id: str, filename: str, content: bytes) -> Path:
        project_id = validate_urn_identifier(project_id)
        if not self.store.get_node(project_id, NodeLabel.PROJECT.value):
            raise ValueError("Project does not exist.")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds the 25 MB upload limit.")

        safe_name = _safe_filename(filename)
        if Path(safe_name).suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_DOC_EXTENSIONS))
            raise ValueError(f"Unsupported document type. Supported: {supported}")

        project_dir = self.upload_root / project_id.rsplit(":", 1)[-1]
        project_dir.mkdir(parents=True, exist_ok=True)
        destination = project_dir / safe_name
        destination.write_bytes(content)
        return project_dir

    def ingest_documents(self, project_id: str, documents_dir: Path) -> dict[str, Any]:
        summary = ingest_documents(
            str(documents_dir),
            project_urn=project_id,
            company_urn=f"urn:company:local:{project_id.rsplit(':', 1)[-1]}",
            store=self.store,
        )
        return summary.to_dict()
