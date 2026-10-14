"""
Bounded Symbol Resolution and Relationship Extractor (Task 4 & Task 5).
Resolves imports, function calls, class inheritance, API implementations, and tests.
Honoring strict provenance: unresolved symbols are recorded honestly, not speculated.
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from app.graph.schema import (
    NodeLabel,
    RelationshipType,
    ProvenanceType,
)
from app.graph.ingestion.code.identifiers import (
    build_package_urn,
    build_file_urn,
    build_test_urn,
)
from app.graph.ingestion.code.parser import ParseResult, ParsedFunction, ParsedClass, ParsedCall


@dataclass
class ResolvedRelationship:
    rel_type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnresolvedReference:
    caller_urn: str
    callee_expression: str
    line_number: int
    source_file: str
    reason: str


@dataclass
class ResolutionContext:
    repo_urn: str
    parse_results: List[ParseResult]

    # Quick lookup indices
    file_map: Dict[str, ParseResult] = field(default_factory=dict)         # rel_path -> ParseResult
    file_urn_map: Dict[str, str] = field(default_factory=dict)             # rel_path -> file_urn
    func_map: Dict[str, ParsedFunction] = field(default_factory=dict)      # func_urn -> ParsedFunction
    class_map: Dict[str, ParsedClass] = field(default_factory=dict)        # class_urn -> ParsedClass

    # Module name to file mapping: e.g. "app.config" -> "app/config.py"
    module_to_file: Dict[str, str] = field(default_factory=dict)

    # Symbol name lookup by file: (file_urn, simple_name) -> target_urn
    file_symbols: Dict[Tuple[str, str], str] = field(default_factory=dict)


class SymbolResolver:
    """
    Performs bounded static resolution across all parsed files in a repository.
    """

    def __init__(self, repo_urn: str, parse_results: List[ParseResult]):
        self.ctx = ResolutionContext(repo_urn=repo_urn, parse_results=parse_results)
        self.relationships: List[ResolvedRelationship] = []
        self.unresolved: List[UnresolvedReference] = []

        self._build_symbol_tables()

    def _build_symbol_tables(self):
        """Constructs lookup indexes for fast symbol resolution."""
        for pr in self.ctx.parse_results:
            if not pr.success:
                continue
            self.ctx.file_map[pr.rel_path] = pr
            self.ctx.file_urn_map[pr.rel_path] = pr.file_urn

            # Map Python dotted module paths to relative file paths
            # e.g. "app/agent/graph.py" -> "app.agent.graph"
            clean_rel = pr.rel_path.removesuffix(".py").replace("\\", "/")
            dotted_mod = clean_rel.replace("/", ".")
            self.ctx.module_to_file[dotted_mod] = pr.rel_path
            if dotted_mod.startswith("src."):
                # Support the conventional src/ layout, whose imports omit src.
                self.ctx.module_to_file[dotted_mod.removeprefix("src.")] = pr.rel_path
            if clean_rel.endswith("__init__"):
                parent_mod = dotted_mod.removesuffix(".__init__")
                self.ctx.module_to_file[parent_mod] = pr.rel_path

            # Register classes
            for c in pr.classes:
                self.ctx.class_map[c.urn] = c
                self.ctx.file_symbols[(pr.file_urn, c.name)] = c.urn

            # Register functions
            for f in pr.functions:
                self.ctx.func_map[f.urn] = f
                self.ctx.file_symbols[(pr.file_urn, f.name)] = f.urn

    def _resolve_import_module(self, pr: ParseResult, imp) -> Optional[str]:
        """Resolve absolute and relative imports to a verified scanned module."""
        module = imp.module
        if imp.level:
            current = pr.rel_path.removesuffix(".py").replace("/", ".")
            package = current.split(".") if current.endswith(".__init__") else current.split(".")[:-1]
            ascend = imp.level - 1
            if ascend > len(package):
                return None
            base = package[:len(package) - ascend]
            module = ".".join(base + ([imp.module] if imp.module else []))
        return self.ctx.module_to_file.get(module)

    def resolve_all(self) -> Tuple[List[ResolvedRelationship], List[UnresolvedReference]]:
        """Executes resolution for containment, imports, inheritance, calls, APIs, and tests."""
        for pr in self.ctx.parse_results:
            if not pr.success:
                continue

            self._resolve_containment(pr)
            self._resolve_imports(pr)
            self._resolve_inheritance(pr)
            self._resolve_calls(pr)
            self._resolve_apis(pr)
            self._resolve_tests(pr)

        return self.relationships, self.unresolved

    # ── 1. Containment Relationships (Repo -> File -> Class/Function) ──────────

    def _resolve_containment(self, pr: ParseResult):
        # Repo CONTAINS File
        self.relationships.append(
            ResolvedRelationship(
                rel_type=RelationshipType.CONTAINS.value,
                source_id=self.ctx.repo_urn,
                target_id=pr.file_urn,
                properties={"loc": pr.loc},
                provenance={
                    "provenance_type": ProvenanceType.STATIC_AST.value,
                    "confidence": 1.0,
                    "source_file_path": pr.rel_path,
                },
            )
        )

        # File CONTAINS Classes
        for c in pr.classes:
            parent_id = c.parent_class_urn or pr.file_urn
            self.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.CONTAINS.value,
                    source_id=parent_id,
                    target_id=c.urn,
                    properties={"line_start": c.line_start, "line_end": c.line_end},
                    provenance={
                        "provenance_type": ProvenanceType.STATIC_AST.value,
                        "confidence": 1.0,
                        "source_file_path": pr.rel_path,
                    },
                )
            )

        # File/Class CONTAINS Functions
        for f in pr.functions:
            parent_id = f.parent_class_urn or pr.file_urn
            self.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.CONTAINS.value,
                    source_id=parent_id,
                    target_id=f.urn,
                    properties={"line_start": f.line_start, "line_end": f.line_end},
                    provenance={
                        "provenance_type": ProvenanceType.STATIC_AST.value,
                        "confidence": 1.0,
                        "source_file_path": pr.rel_path,
                    },
                )
            )

    # ── 2. Imports ─────────────────────────────────────────────────────────────

    def _resolve_imports(self, pr: ParseResult):
        for imp in pr.imports:
            # Check if this import refers to an internal module
            target_rel_path = self._resolve_import_module(pr, imp)
            if target_rel_path:
                target_file_urn = self.ctx.file_urn_map[target_rel_path]
                self.relationships.append(
                    ResolvedRelationship(
                        rel_type=RelationshipType.IMPORTS.value,
                        source_id=pr.file_urn,
                        target_id=target_file_urn,
                        properties={
                            "line_number": imp.line_number,
                            "imported_name": imp.imported_name,
                            "alias": imp.alias,
                            "relative_level": imp.level,
                        },
                        provenance={
                            "provenance_type": ProvenanceType.STATIC_AST.value,
                            "confidence": 1.0,
                            "source_file_path": pr.rel_path,
                        },
                    )
                )
            else:
                # External third-party package
                pkg_name = imp.module.split(".")[0]
                if pkg_name:
                    pkg_urn = build_package_urn(pkg_name)
                    self.relationships.append(
                        ResolvedRelationship(
                            rel_type=RelationshipType.IMPORTS.value,
                            source_id=pr.file_urn,
                            target_id=pkg_urn,
                            properties={"line_number": imp.line_number, "external": True},
                            provenance={
                                "provenance_type": ProvenanceType.STATIC_AST.value,
                                "confidence": 1.0,
                                "source_file_path": pr.rel_path,
                            },
                        )
                    )

    # ── 3. Class Inheritance ───────────────────────────────────────────────────

    def _resolve_inheritance(self, pr: ParseResult):
        for c in pr.classes:
            for base_name in c.base_classes:
                # 1. Look up in same file
                target_urn = self.ctx.file_symbols.get((pr.file_urn, base_name))

                # 2. Look up in imported symbols
                if not target_urn:
                    for imp in pr.imports:
                        if imp.imported_name == base_name or (not imp.imported_name and imp.alias == base_name):
                            target_rel = self._resolve_import_module(pr, imp)
                            if target_rel:
                                target_file_urn = self.ctx.file_urn_map[target_rel]
                                target_urn = self.ctx.file_symbols.get((target_file_urn, base_name))
                            break

                if target_urn and target_urn in self.ctx.class_map:
                    self.relationships.append(
                        ResolvedRelationship(
                            rel_type=RelationshipType.INHERITS.value,
                            source_id=c.urn,
                            target_id=target_urn,
                            properties={"line_number": c.line_start},
                            provenance={
                                "provenance_type": ProvenanceType.STATIC_AST.value,
                                "confidence": 1.0,
                                "source_file_path": pr.rel_path,
                            },
                        )
                    )

    # ── 4. Function Calls (Bounded Resolution) ─────────────────────────────────

    def _resolve_calls(self, pr: ParseResult):
        for call in pr.calls:
            expr = call.callee_expression
            target_urn = None

            # Case A: Method call on self (e.g. self.verify_connectivity)
            if expr.startswith("self.") and call.caller_urn in self.ctx.func_map:
                method_name = expr.split("self.", 1)[1]
                caller_func = self.ctx.func_map[call.caller_urn]
                if caller_func.parent_class_urn:
                    parent_class = self.ctx.class_map.get(caller_func.parent_class_urn)
                    if parent_class:
                        target_urn = self.ctx.file_symbols.get(
                            (pr.file_urn, f"{parent_class.name}.{method_name}")
                        ) or self.ctx.file_symbols.get((pr.file_urn, method_name))

            # Case B: Local function call within the same file
            if not target_urn and "." not in expr:
                target_urn = self.ctx.file_symbols.get((pr.file_urn, expr))

            # Case C: Call to directly imported function (e.g. from x import y; y())
            if not target_urn and "." not in expr:
                for imp in pr.imports:
                    if imp.imported_name == expr or imp.alias == expr:
                        target_rel = self._resolve_import_module(pr, imp)
                        if target_rel:
                            target_file_urn = self.ctx.file_urn_map[target_rel]
                            target_urn = self.ctx.file_symbols.get((target_file_urn, imp.imported_name or expr))
                        break

            # Case D: Module-qualified call (e.g. qdrant_service.search_enterprise_knowledge())
            if not target_urn and "." in expr:
                parts = expr.split(".")
                mod_alias, func_name = parts[0], parts[-1]
                for imp in pr.imports:
                    if (imp.alias and imp.alias == mod_alias) or (not imp.alias and imp.module.endswith(mod_alias)):
                        target_rel = self._resolve_import_module(pr, imp)
                        if target_rel:
                            target_file_urn = self.ctx.file_urn_map[target_rel]
                            target_urn = self.ctx.file_symbols.get((target_file_urn, func_name))
                        break

            # If resolved to a verified function in our graph
            if target_urn and target_urn in self.ctx.func_map:
                self.relationships.append(
                    ResolvedRelationship(
                        rel_type=RelationshipType.CALLS.value,
                        source_id=call.caller_urn,
                        target_id=target_urn,
                        properties={"line_number": call.line_number},
                        provenance={
                            "provenance_type": ProvenanceType.STATIC_AST.value,
                            "confidence": 1.0,
                            "source_file_path": pr.rel_path,
                            "evidence_snippet": f"CALLS {expr}",
                        },
                    )
                )
            else:
                # Record honestly as unresolved
                self.unresolved.append(
                    UnresolvedReference(
                        caller_urn=call.caller_urn,
                        callee_expression=expr,
                        line_number=call.line_number,
                        source_file=pr.rel_path,
                        reason="Target not resolved in static module scope",
                    )
                )

    # ── 5. API Implementations & Exposure ──────────────────────────────────────

    def _resolve_apis(self, pr: ParseResult):
        for api in pr.apis:
            # Handler IMPLEMENTS API
            self.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.IMPLEMENTS.value,
                    source_id=api.handler_func_urn,
                    target_id=api.urn,
                    properties={"framework": api.framework, "line_number": api.line_number},
                    provenance={
                        "provenance_type": ProvenanceType.STATIC_AST.value,
                        "confidence": 1.0,
                        "source_file_path": pr.rel_path,
                    },
                )
            )

            # Repo/Service EXPOSES API
            self.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.EXPOSES.value,
                    source_id=self.ctx.repo_urn,
                    target_id=api.urn,
                    properties={"route": api.route, "method": api.http_method},
                    provenance={
                        "provenance_type": ProvenanceType.STATIC_AST.value,
                        "confidence": 1.0,
                        "source_file_path": pr.rel_path,
                    },
                )
            )

    # ── 6. Test Links (TESTED_BY) ──────────────────────────────────────────────

    def _resolve_tests(self, pr: ParseResult):
        for test_func in pr.tests:
            # Rule: If test function name is test_<target>, find <target>
            if test_func.name.startswith("test_"):
                target_candidate = test_func.name.removeprefix("test_")
                # Look for matching function across codebase
                for (f_urn, f_name), candidate_urn in self.ctx.file_symbols.items():
                    if f_name == target_candidate and candidate_urn in self.ctx.func_map:
                        self.relationships.append(
                            ResolvedRelationship(
                                rel_type=RelationshipType.TESTED_BY.value,
                                source_id=candidate_urn,
                                target_id=test_func.urn,
                                properties={"rule": "name_matching_convention"},
                                provenance={
                                    "provenance_type": ProvenanceType.HEURISTIC.value,
                                    "confidence": 0.85,
                                    "source_file_path": pr.rel_path,
                                },
                            )
                        )
                        break
