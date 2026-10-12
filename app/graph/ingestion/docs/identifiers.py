"""
Deterministic URN Generator for Document and Policy Entities (Phase 01 / Phase 04).
Generates stable identifiers distinguishing documents, architecture components,
policies, security rules, and coding standards.
"""

import re
from app.graph.schema import validate_urn_identifier
from app.graph.ingestion.code.identifiers import normalize_relative_path


def build_doc_urn(project_or_repo_urn: str, rel_path: str) -> str:
    """urn:doc:<scope>:<normalized_rel_path>"""
    scope = project_or_repo_urn.removeprefix("urn:project:").removeprefix("urn:repo:").removeprefix("urn:")
    norm_path = normalize_relative_path(rel_path)
    urn = f"urn:doc:{scope}:{norm_path}"
    return validate_urn_identifier(urn)


def build_policy_urn(company_urn: str, policy_code: str) -> str:
    """urn:policy:<company_scope>:<policy_code>"""
    scope = company_urn.removeprefix("urn:company:").removeprefix("urn:")
    clean_code = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", policy_code.strip())
    urn = f"urn:policy:{scope}:{clean_code}"
    return validate_urn_identifier(urn)


def build_security_rule_urn(policy_urn: str, rule_slug: str) -> str:
    """urn:secrule:<policy_scope>:<rule_slug>"""
    scope = policy_urn.removeprefix("urn:policy:").removeprefix("urn:")
    clean_slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", rule_slug.strip().lower())
    urn = f"urn:secrule:{scope}:{clean_slug}"
    return validate_urn_identifier(urn)


def build_coding_standard_urn(company_urn: str, standard_code: str) -> str:
    """urn:standard:<company_scope>:<standard_code>"""
    scope = company_urn.removeprefix("urn:company:").removeprefix("urn:")
    clean_code = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", standard_code.strip())
    urn = f"urn:standard:{scope}:{clean_code}"
    return validate_urn_identifier(urn)


def build_arch_component_urn(project_urn: str, component_slug: str) -> str:
    """urn:arch:<project_scope>:<component_slug>"""
    scope = project_urn.removeprefix("urn:project:").removeprefix("urn:")
    clean_slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", component_slug.strip().lower())
    urn = f"urn:arch:{scope}:{clean_slug}"
    return validate_urn_identifier(urn)
