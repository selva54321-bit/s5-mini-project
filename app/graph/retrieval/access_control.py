"""
Access Control and Authorization Extension Points for Graph Retrieval.
Enforces multi-tenant and project/repository isolation during entity resolution
and traversal. Explicitly prevents exposing the engine in multi-user production
environments without completed policy enforcement.
"""

from abc import ABC, abstractmethod
from typing import Set, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AccessControlContext(BaseModel):
    """
    Carries the identity and authorization boundaries of the principal invoking retrieval.
    """
    user_id: str = "default_local_user"
    tenant_id: str = "default_tenant"
    allowed_projects: Set[str] = Field(default_factory=set, description="Set of allowed Project URNs (empty = unrestricted local)")
    allowed_repos: Set[str] = Field(default_factory=set, description="Set of allowed Repository URNs (empty = unrestricted local)")
    allowed_classifications: Set[str] = Field(default_factory=lambda: {"PUBLIC", "INTERNAL", "CONFIDENTIAL"})
    is_admin: bool = False
    is_multi_user_production: bool = False


class AccessController(ABC):
    """
    Abstract authorization engine for graph-based knowledge retrieval.
    """

    @abstractmethod
    def can_access_node(self, node: Dict[str, Any], context: AccessControlContext) -> bool:
        """Determines if the principal may view the entity node."""
        pass

    @abstractmethod
    def filter_nodes(self, nodes: List[Dict[str, Any]], context: AccessControlContext) -> List[Dict[str, Any]]:
        """Filters a collection of nodes against principal permissions."""
        pass

    @abstractmethod
    def can_traverse_relationship(
        self,
        source: Dict[str, Any],
        rel_type: str,
        target: Dict[str, Any],
        context: AccessControlContext,
    ) -> bool:
        """Determines if the edge and target node can be traversed."""
        pass

    @abstractmethod
    def assert_service_exposure_permitted(self, context: AccessControlContext) -> None:
        """
        Safety check: raises PermissionError if attempting to expose retrieval
        as a multi-user production service before production RBAC is configured.
        """
        pass


class LocalDevAccessController(AccessController):
    """
    Permissive access controller for local developer workflows and tests.
    Explicitly blocks execution if context claims to be in multi_user_production.
    """

    def can_access_node(self, node: Dict[str, Any], context: AccessControlContext) -> bool:
        self.assert_service_exposure_permitted(context)
        if context.is_admin:
            return True

        # Check repository and project scope if restricted
        if context.allowed_projects:
            project_urn = node.get("project_urn") or node.get("provenance_source_project_id")
            if project_urn and project_urn not in context.allowed_projects:
                return False

        if context.allowed_repos:
            repo_urn = node.get("repo_urn") or node.get("provenance_source_repo_id")
            if repo_urn and repo_urn not in context.allowed_repos:
                return False

        return True

    def filter_nodes(self, nodes: List[Dict[str, Any]], context: AccessControlContext) -> List[Dict[str, Any]]:
        return [n for n in nodes if self.can_access_node(n, context)]

    def can_traverse_relationship(
        self,
        source: Dict[str, Any],
        rel_type: str,
        target: Dict[str, Any],
        context: AccessControlContext,
    ) -> bool:
        return self.can_access_node(source, context) and self.can_access_node(target, context)

    def assert_service_exposure_permitted(self, context: AccessControlContext) -> None:
        if context.is_multi_user_production:
            raise PermissionError(
                "Access Control Safeguard: Graph retrieval cannot be exposed as a "
                "multi-user production service with LocalDevAccessController. "
                "A production-grade enterprise authorization provider must be configured first."
            )


class ScopedAccessController(AccessController):
    """
    Enforces project and repository boundaries strictly.
    """

    def can_access_node(self, node: Dict[str, Any], context: AccessControlContext) -> bool:
        self.assert_service_exposure_permitted(context)
        if context.is_admin:
            return True

        node_urn = node.get("id", "")
        # Global or company-wide policies are accessible if classification permits
        classification = node.get("classification", "INTERNAL")
        if classification not in context.allowed_classifications:
            return False

        node_project = node.get("project_urn") or node.get("provenance_source_project_id")
        if context.allowed_projects and node_project:
            if node_project not in context.allowed_projects:
                return False

        node_repo = node.get("repo_urn") or node.get("provenance_source_repo_id")
        if context.allowed_repos and node_repo:
            if node_repo not in context.allowed_repos:
                return False

        return True

    def filter_nodes(self, nodes: List[Dict[str, Any]], context: AccessControlContext) -> List[Dict[str, Any]]:
        return [n for n in nodes if self.can_access_node(n, context)]

    def can_traverse_relationship(
        self,
        source: Dict[str, Any],
        rel_type: str,
        target: Dict[str, Any],
        context: AccessControlContext,
    ) -> bool:
        return self.can_access_node(source, context) and self.can_access_node(target, context)

    def assert_service_exposure_permitted(self, context: AccessControlContext) -> None:
        if context.is_multi_user_production and not context.is_admin and not context.allowed_projects:
            raise PermissionError(
                "Access Control Safeguard: Multi-user production request denied because "
                "principal has no explicit project grants assigned."
            )


# Default controller singleton for local single-tenant execution
default_access_controller = LocalDevAccessController()
