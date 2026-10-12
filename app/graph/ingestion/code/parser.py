"""
Python AST Code Parser (Task 2).
Uses Python's built-in ast module to parse source code deterministically without execution.
Extracts File, Class, Function, API endpoints, Test cases, Imports, and Call expressions.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from app.graph.ingestion.code.identifiers import (
    build_file_urn,
    build_class_urn,
    build_function_urn,
    build_api_urn,
    build_test_urn,
)


@dataclass
class ParsedImport:
    module: str
    imported_name: Optional[str]
    alias: Optional[str]
    is_from_import: bool
    line_number: int


@dataclass
class ParsedCall:
    caller_urn: str
    callee_expression: str
    line_number: int


@dataclass
class ParsedAPI:
    urn: str
    route: str
    http_method: str
    handler_func_urn: str
    framework: str
    line_number: int


@dataclass
class ParsedFunction:
    urn: str
    name: str
    qualified_name: str
    file_urn: str
    parent_class_urn: Optional[str]
    is_method: bool
    is_async: bool
    line_start: int
    line_end: int
    parameters: List[str]
    docstring: Optional[str]
    decorators: List[str]
    is_test: bool


@dataclass
class ParsedClass:
    urn: str
    name: str
    qualified_name: str
    file_urn: str
    parent_class_urn: Optional[str]
    base_classes: List[str]
    line_start: int
    line_end: int
    docstring: Optional[str]
    decorators: List[str]
    is_test: bool


@dataclass
class ParseResult:
    rel_path: str
    file_urn: str
    success: bool
    loc: int = 0
    classes: List[ParsedClass] = field(default_factory=list)
    functions: List[ParsedFunction] = field(default_factory=list)
    imports: List[ParsedImport] = field(default_factory=list)
    calls: List[ParsedCall] = field(default_factory=list)
    apis: List[ParsedAPI] = field(default_factory=list)
    tests: List[ParsedFunction] = field(default_factory=list)
    db_operations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


class CodeASTVisitor(ast.NodeVisitor):
    """
    AST visitor that traverses a Python module to extract structured entities.
    Tracks lexical scope stacks (classes, functions) to produce fully qualified symbol paths.
    """

    def __init__(self, file_urn: str, repo_urn: str, rel_path: str):
        self.file_urn = file_urn
        self.repo_urn = repo_urn
        self.rel_path = rel_path

        self.scope_stack: List[str] = []
        self.class_urn_stack: List[str] = []

        self.classes: List[ParsedClass] = []
        self.functions: List[ParsedFunction] = []
        self.imports: List[ParsedImport] = []
        self.calls: List[ParsedCall] = []
        self.apis: List[ParsedAPI] = []
        self.tests: List[ParsedFunction] = []
        self.db_operations: List[Dict[str, Any]] = []

    def current_caller_urn(self) -> str:
        """Returns the URN of the currently executing function, or file URN for module-level."""
        if self.scope_stack:
            return build_function_urn(self.file_urn, ".".join(self.scope_stack))
        return self.file_urn

    # ── Imports ───────────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    module=alias.name,
                    imported_name=None,
                    alias=alias.asname,
                    is_from_import=False,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    module=module,
                    imported_name=alias.name,
                    alias=alias.asname,
                    is_from_import=True,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    # ── Classes ───────────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        self.scope_stack.append(class_name)
        qname = ".".join(self.scope_stack)
        class_urn = build_class_urn(self.file_urn, qname)

        parent_class_urn = self.class_urn_stack[-1] if self.class_urn_stack else None
        self.class_urn_stack.append(class_urn)

        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        decorators = [self._decorator_to_str(d) for d in node.decorator_list]
        is_test = class_name.startswith("Test") or "test" in self.rel_path.lower()

        parsed_class = ParsedClass(
            urn=class_urn,
            name=class_name,
            qualified_name=qname,
            file_urn=self.file_urn,
            parent_class_urn=parent_class_urn,
            base_classes=bases,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
            decorators=decorators,
            is_test=is_test,
        )
        self.classes.append(parsed_class)

        self.generic_visit(node)

        self.class_urn_stack.pop()
        self.scope_stack.pop()

    # ── Functions & Methods ───────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node, is_async=True)

    def _process_function(self, node, is_async: bool):
        func_name = node.name
        self.scope_stack.append(func_name)
        qname = ".".join(self.scope_stack)
        func_urn = build_function_urn(self.file_urn, qname)

        is_method = bool(self.class_urn_stack)
        parent_class_urn = self.class_urn_stack[-1] if is_method else None

        decorators = [self._decorator_to_str(d) for d in node.decorator_list]
        params = [arg.arg for arg in node.args.args]
        is_test = func_name.startswith("test_") or "test" in self.rel_path.lower()

        parsed_func = ParsedFunction(
            urn=func_urn,
            name=func_name,
            qualified_name=qname,
            file_urn=self.file_urn,
            parent_class_urn=parent_class_urn,
            is_method=is_method,
            is_async=is_async,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            parameters=params,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            is_test=is_test,
        )
        self.functions.append(parsed_func)

        if is_test:
            self.tests.append(parsed_func)

        # Detect FastAPI / route decorators
        self._detect_api_route(node, func_urn)

        self.generic_visit(node)

        self.scope_stack.pop()

    # ── Function Calls ────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call):
        callee_expr = self._call_to_str(node.func)
        if callee_expr:
            caller_urn = self.current_caller_urn()
            self.calls.append(
                ParsedCall(
                    caller_urn=caller_urn,
                    callee_expression=callee_expr,
                    line_number=node.lineno,
                )
            )

            # Detect known database access expressions
            if any(term in callee_expr for term in ("query_points", "upsert", "execute_write", "execute_read", "execute")):
                self.db_operations.append({
                    "caller_urn": caller_urn,
                    "expression": callee_expr,
                    "line_number": node.lineno,
                })

        self.generic_visit(node)

    # ── Helper Extractors ─────────────────────────────────────────────────────

    def _detect_api_route(self, node, func_urn: str):
        """Detects @app.get('/route'), @router.post('/route'), etc."""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                attr_name = ""
                if isinstance(dec.func, ast.Attribute):
                    attr_name = dec.func.attr.upper()
                elif isinstance(dec.func, ast.Name):
                    attr_name = dec.func.id.upper()

                if attr_name in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
                    # Check first argument for route string
                    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                        route_path = dec.args[0].value
                        api_urn = build_api_urn(self.repo_urn, attr_name, route_path)
                        self.apis.append(
                            ParsedAPI(
                                urn=api_urn,
                                route=route_path,
                                http_method=attr_name,
                                handler_func_urn=func_urn,
                                framework="FastAPI",
                                line_number=node.lineno,
                            )
                        )

    def _decorator_to_str(self, dec) -> str:
        if isinstance(dec, ast.Name):
            return dec.id
        elif isinstance(dec, ast.Attribute):
            return self._get_attribute_name(dec)
        elif isinstance(dec, ast.Call):
            func_name = self._decorator_to_str(dec.func)
            return f"{func_name}(...)"
        return "decorator"

    def _call_to_str(self, func_node) -> str:
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return self._get_attribute_name(func_node)
        return ""

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        parts.reverse()
        return ".".join(parts)


def parse_python_file(abs_path: str, rel_path: str, repo_urn: str) -> ParseResult:
    """
    Parses a single Python file using AST.
    Safely captures syntax errors without throwing unhandled exceptions.
    """
    file_urn = build_file_urn(repo_urn, rel_path)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
    except Exception as e:
        return ParseResult(
            rel_path=rel_path,
            file_urn=file_urn,
            success=False,
            error_message=f"File read error: {e}",
        )

    loc = len(code.splitlines())

    try:
        tree = ast.parse(code, filename=rel_path)
    except SyntaxError as e:
        return ParseResult(
            rel_path=rel_path,
            file_urn=file_urn,
            success=False,
            loc=loc,
            error_message=f"SyntaxError on line {e.lineno}: {e.msg}",
        )
    except Exception as e:
        return ParseResult(
            rel_path=rel_path,
            file_urn=file_urn,
            success=False,
            loc=loc,
            error_message=f"AST parse error: {e}",
        )

    visitor = CodeASTVisitor(file_urn=file_urn, repo_urn=repo_urn, rel_path=rel_path)
    visitor.visit(tree)

    return ParseResult(
        rel_path=rel_path,
        file_urn=file_urn,
        success=True,
        loc=loc,
        classes=visitor.classes,
        functions=visitor.functions,
        imports=visitor.imports,
        calls=visitor.calls,
        apis=visitor.apis,
        tests=visitor.tests,
        db_operations=visitor.db_operations,
    )
