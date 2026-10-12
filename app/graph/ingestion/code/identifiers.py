"""
Deterministic URN Generator for Code Entities (Phase 01 / Phase 03).
Generates stable identifiers distinguishing symbols across repositories, files, classes, and scopes.
"""

import re
from app.graph.schema import validate_urn_identifier


def normalize_relative_path(path: str) -> str:
    """Normalize file paths using forward slashes and removing leading slashes/dots."""
    norm = path.replace("\\", "/").strip()
    norm = re.sub(r"^\./", "", norm)
    return norm.lstrip("/")


def normalize_route(route: str) -> str:
    """Normalize an API route path."""
    cleaned = route.strip()
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    # Normalize trailing slash except root
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned


def build_repo_urn(repo_name: str, org: str = "local", provider: str = "git") -> str:
    """urn:repo:<provider>:<org>:<repo_name>"""
    clean_repo = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", repo_name.strip())
    clean_org = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", org.strip())
    urn = f"urn:repo:{provider}:{clean_org}:{clean_repo}"
    return validate_urn_identifier(urn)


def build_file_urn(repo_urn: str, rel_path: str) -> str:
    """urn:file:<repo_scope>:<relative_path>"""
    repo_scope = repo_urn.removeprefix("urn:repo:").removeprefix("urn:")
    norm_path = normalize_relative_path(rel_path)
    urn = f"urn:file:{repo_scope}:{norm_path}"
    return validate_urn_identifier(urn)


def build_class_urn(file_urn: str, qualified_class_name: str) -> str:
    """urn:class:<file_scope>:<qualified_class_name>"""
    file_scope = file_urn.removeprefix("urn:file:").removeprefix("urn:")
    urn = f"urn:class:{file_scope}:{qualified_class_name.strip()}"
    return validate_urn_identifier(urn)


def build_function_urn(file_urn: str, qualified_func_name: str) -> str:
    """urn:func:<file_scope>:<qualified_function_name>"""
    file_scope = file_urn.removeprefix("urn:file:").removeprefix("urn:")
    urn = f"urn:func:{file_scope}:{qualified_func_name.strip()}"
    return validate_urn_identifier(urn)


def build_api_urn(service_or_repo_urn: str, http_method: str, route: str) -> str:
    """urn:api:<scope>:<HTTP_METHOD>:<normalized_route>"""
    scope = service_or_repo_urn.removeprefix("urn:service:").removeprefix("urn:repo:").removeprefix("urn:")
    method = http_method.strip().upper()
    norm_route = normalize_route(route)
    urn = f"urn:api:{scope}:{method}:{norm_route}"
    return validate_urn_identifier(urn)


def build_test_urn(file_urn: str, test_name: str) -> str:
    """urn:test:<file_scope>:<test_name>"""
    file_scope = file_urn.removeprefix("urn:file:").removeprefix("urn:")
    urn = f"urn:test:{file_scope}:{test_name.strip()}"
    return validate_urn_identifier(urn)


def build_package_urn(package_name: str, ecosystem: str = "pypi") -> str:
    """urn:pkg:<ecosystem>:<package_name>"""
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", package_name.strip().lower())
    urn = f"urn:pkg:{ecosystem}:{clean_name}"
    return validate_urn_identifier(urn)
