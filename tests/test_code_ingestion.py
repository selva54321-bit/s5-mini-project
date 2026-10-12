"""
Comprehensive Test Suite for Code Analysis & Repository Ingestion (Phase 03).
Tests repository discovery, AST parsing, symbol resolution, API extraction,
unresolved reference tracking, syntax error handling, and pipeline execution.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from app.graph.schema import NodeLabel, RelationshipType
from app.graph.ingestion.code.identifiers import (
    build_repo_urn,
    build_file_urn,
    build_class_urn,
    build_function_urn,
    build_api_urn,
    build_test_urn,
    build_package_urn,
)
from app.graph.ingestion.code.scanner import scan_repository, is_path_excluded
from app.graph.ingestion.code.parser import parse_python_file
from app.graph.ingestion.code.resolver import SymbolResolver
from app.graph.ingestion.code.pipeline import ingest_repository


# ── 1. Identifier Generation Tests ────────────────────────────────────────────

def test_stable_identifiers_distinguish_duplicate_names():
    repo_urn = build_repo_urn("my-project", org="acme")
    file1_urn = build_file_urn(repo_urn, "services/auth.py")
    file2_urn = build_file_urn(repo_urn, "services/payment.py")

    # Identical function name in different files
    fn1_urn = build_function_urn(file1_urn, "process")
    fn2_urn = build_function_urn(file2_urn, "process")

    assert fn1_urn != fn2_urn
    assert "services/auth.py:process" in fn1_urn
    assert "services/payment.py:process" in fn2_urn


def test_stable_identifiers_distinguish_class_methods():
    repo_urn = build_repo_urn("my-project")
    file_urn = build_file_urn(repo_urn, "app/models.py")

    class1_method = build_function_urn(file_urn, "User.save")
    class2_method = build_function_urn(file_urn, "Order.save")

    assert class1_method != class2_method
    assert "User.save" in class1_method
    assert "Order.save" in class2_method


def test_api_and_test_identifiers():
    repo_urn = build_repo_urn("my-project")
    file_urn = build_file_urn(repo_urn, "tests/test_auth.py")

    api_urn = build_api_urn(repo_urn, "post", "/api/v1/login")
    assert api_urn == "urn:api:git:local:my-project:POST:/api/v1/login"

    test_urn = build_test_urn(file_urn, "test_login_success")
    assert "test_login_success" in test_urn


# ── 2. Repository Discovery & Exclusion Tests ─────────────────────────────────

def test_scanner_exclusion_rules():
    assert is_path_excluded(".git", ".git", is_dir=True) is True
    assert is_path_excluded(".venv", ".venv", is_dir=True) is True
    assert is_path_excluded("__pycache__", "__pycache__", is_dir=True) is True
    assert is_path_excluded("file.pyc", "file.pyc", is_dir=False) is True
    assert is_path_excluded("src/main.py", "main.py", is_dir=False) is False


def test_scanner_traversal_on_temp_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create structure:
        # repo/
        #   app/
        #     __init__.py
        #     main.py
        #   .venv/
        #     lib.py (should be skipped)
        #   ignored.pyc (should be skipped)
        app_dir = os.path.join(tmpdir, "app")
        venv_dir = os.path.join(tmpdir, ".venv")
        os.makedirs(app_dir)
        os.makedirs(venv_dir)

        with open(os.path.join(app_dir, "__init__.py"), "w") as f:
            f.write("# init")
        with open(os.path.join(app_dir, "main.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(venv_dir, "lib.py"), "w") as f:
            f.write("# excluded")
        with open(os.path.join(tmpdir, "test.pyc"), "w") as f:
            f.write("bytecode")

        discovered = scan_repository(tmpdir)
        rel_paths = [d.rel_path for d in discovered]

        assert "app/__init__.py" in rel_paths
        assert "app/main.py" in rel_paths
        assert not any(".venv" in p for p in rel_paths)
        assert not any(".pyc" in p for p in rel_paths)


# ── 3. AST Parser Tests ───────────────────────────────────────────────────────

SAMPLE_PYTHON_CODE = """
import os
from math import sqrt
import requests as req

class BaseService:
    def execute(self):
        return True

class UserService(BaseService):
    '''User domain service.'''
    def __init__(self, db_client):
        self.db = db_client

    def find_user(self, user_id: str):
        result = self.execute()
        return result

def helper_func(x):
    return sqrt(x)

@app.post("/users")
def create_user_api():
    return helper_func(42)

def test_find_user():
    service = UserService(None)
    assert service.execute() is True
"""

def test_ast_parser_extracts_entities():
    repo_urn = build_repo_urn("test-repo")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SAMPLE_PYTHON_CODE)
        temp_path = f.name

    try:
        res = parse_python_file(abs_path=temp_path, rel_path="services/user.py", repo_urn=repo_urn)
        assert res.success is True
        assert res.loc > 20

        # Verify Classes
        class_names = [c.name for c in res.classes]
        assert "BaseService" in class_names
        assert "UserService" in class_names

        # Verify Inheritance base class
        user_service = next(c for c in res.classes if c.name == "UserService")
        assert "BaseService" in user_service.base_classes

        # Verify Functions and Methods
        func_qnames = [fn.qualified_name for fn in res.functions]
        assert "BaseService.execute" in func_qnames
        assert "UserService.find_user" in func_qnames
        assert "helper_func" in func_qnames
        assert "create_user_api" in func_qnames
        assert "test_find_user" in func_qnames

        # Verify Imports
        imported_mods = [imp.module for imp in res.imports]
        assert "os" in imported_mods
        assert "math" in imported_mods
        assert "requests" in imported_mods

        # Verify API route extraction
        assert len(res.apis) == 1
        assert res.apis[0].route == "/users"
        assert res.apis[0].http_method == "POST"

        # Verify Test function extraction
        assert len(res.tests) == 1
        assert res.tests[0].name == "test_find_user"

    finally:
        os.remove(temp_path)


def test_ast_parser_handles_syntax_error():
    repo_urn = build_repo_urn("test-repo")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("def broken_syntax(:\n    invalid code")
        temp_path = f.name

    try:
        res = parse_python_file(abs_path=temp_path, rel_path="broken.py", repo_urn=repo_urn)
        assert res.success is False
        assert "SyntaxError" in res.error_message
    finally:
        os.remove(temp_path)


# ── 4. Symbol Resolution Tests ────────────────────────────────────────────────

def test_symbol_resolver_resolution():
    repo_urn = build_repo_urn("test-repo")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SAMPLE_PYTHON_CODE)
        temp_path = f.name

    try:
        parse_res = parse_python_file(abs_path=temp_path, rel_path="services/user.py", repo_urn=repo_urn)
        resolver = SymbolResolver(repo_urn=repo_urn, parse_results=[parse_res])
        relationships, unresolved = resolver.resolve_all()

        rel_types = [r.rel_type for r in relationships]
        assert RelationshipType.CONTAINS.value in rel_types
        assert RelationshipType.INHERITS.value in rel_types
        assert RelationshipType.CALLS.value in rel_types
        assert RelationshipType.IMPLEMENTS.value in rel_types
        assert RelationshipType.EXPOSES.value in rel_types
        assert RelationshipType.TESTED_BY.value in rel_types

        # Verify INHERITS link: UserService -> BaseService
        inherits_rel = next(r for r in relationships if r.rel_type == RelationshipType.INHERITS.value)
        assert "UserService" in inherits_rel.source_id
        assert "BaseService" in inherits_rel.target_id

        # Verify CALLS link: create_user_api -> helper_func
        call_rel = next(
            r for r in relationships
            if r.rel_type == RelationshipType.CALLS.value and "create_user_api" in r.source_id
        )
        assert "helper_func" in call_rel.target_id

        # Verify TESTED_BY heuristic link: find_user is TESTED_BY test_find_user
        test_rel = next(r for r in relationships if r.rel_type == RelationshipType.TESTED_BY.value)
        assert "find_user" in test_rel.source_id
        assert "test_find_user" in test_rel.target_id

    finally:
        os.remove(temp_path)


# ── 5. End-to-End Pipeline Dry-Run Tests ──────────────────────────────────────

def test_pipeline_dry_run_on_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small synthetic repository
        mod1 = os.path.join(tmpdir, "math_utils.py")
        mod2 = os.path.join(tmpdir, "calculator.py")

        with open(mod1, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")

        with open(mod2, "w") as f:
            f.write("from math_utils import add\n\ndef compute():\n    return add(1, 2)\n")

        # Run pipeline in dry-run mode (no live database required)
        summary = ingest_repository(
            repo_path=tmpdir,
            repo_name="calc-project",
            write_to_db=False,
        )

        assert summary.files_discovered == 2
        assert summary.files_parsed == 2
        assert summary.files_skipped == 0
        assert len(summary.errors) == 0

        # Check extracted entities
        assert summary.nodes_created_or_updated[NodeLabel.FILE.value] == 2
        assert summary.nodes_created_or_updated[NodeLabel.FUNCTION.value] == 2
        assert summary.relationships_created_or_updated[RelationshipType.CALLS.value] >= 1
        assert summary.relationships_created_or_updated[RelationshipType.IMPORTS.value] >= 1


# ── 6. Mock Database Write Execution & Error Handling ─────────────────────────

def test_pipeline_database_write_with_mock_store():
    mock_store = MagicMock()
    mock_store.upsert_nodes_batch.return_value = 5
    mock_store.upsert_relationships_batch.return_value = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "simple.py"), "w") as f:
            f.write("def greet():\n    return 'hello'\n")

        summary = ingest_repository(
            repo_path=tmpdir,
            repo_name="greet-repo",
            write_to_db=True,
            store=mock_store,
        )

        assert summary.files_parsed == 1
        assert len(summary.errors) == 0
        assert mock_store.upsert_nodes_batch.called
        assert mock_store.upsert_relationships_batch.called


def test_pipeline_database_failure_handling():
    mock_store = MagicMock()
    mock_store.upsert_nodes_batch.side_effect = RuntimeError("Database disk full")

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "simple.py"), "w") as f:
            f.write("def greet():\n    return 'hello'\n")

        summary = ingest_repository(
            repo_path=tmpdir,
            repo_name="greet-repo",
            write_to_db=True,
            store=mock_store,
        )

        assert len(summary.errors) == 1
        assert "Database disk full" in summary.errors[0]
