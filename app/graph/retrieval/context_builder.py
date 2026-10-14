"""
Deterministic Context Construction Module for Graph Retrieval.
Transforms structured RetrievalResult subgraphs into clean, deterministic markdown context
suitable for LLM generation or developer inspection.
Includes source locations, doc page refs, provenance confidence tiers,
and clear distinctions between observed facts and heuristic inferences.
Zero vector ranking, zero embeddings.
"""

from typing import List, Dict, Any, Optional
from app.graph.retrieval.subgraph import RetrievalResult, RetrievedNode, RetrievedEdge, PathResult
from app.graph.retrieval.specs import QueryIntent


# Deterministic priority ordering for node labels
LABEL_ORDER = {
    "Service": 1,
    "ArchitectureComponent": 2,
    "API": 3,
    "Policy": 4,
    "SecurityRule": 5,
    "CodingStandard": 6,
    "Document": 7,
    "Database": 8,
    "Table": 9,
    "Class": 10,
    "Function": 11,
    "Test": 12,
    "File": 13,
    "Package": 14,
    "Repository": 15,
    "Project": 16,
}


def _format_evidence_tag(node: RetrievedNode) -> str:
    """Formats provenance and verification evidence tag."""
    parts = []
    if node.provenance_file:
        loc = node.provenance_file
        if node.provenance_lines:
            loc += f":{node.provenance_lines}"
        parts.append(loc)
    elif node.properties.get("doc_page_number"):
        parts.append(f"Page {node.properties['doc_page_number']}")

    p_type = node.provenance_type or "STATIC_AST"
    parts.append(p_type)

    if node.confidence < 1.0:
        parts.append(f"conf={node.confidence:.2f}")

    # Mark observed vs inferred
    if p_type in ("STATIC_AST", "DOC_EXPLICIT", "GIT_HISTORY"):
        status = "Observed"
    else:
        status = "Inferred"
    parts.append(status)

    return f"[{' | '.join(parts)}]"


class GraphContextBuilder:
    """
    Builds structured, deterministically ordered context representations from RetrievalResult.
    """

    def __init__(self, max_characters: int = 12000):
        self.max_characters = max_characters

    def build_context(self, result: RetrievalResult) -> str:
        """
        Builds deterministic context string from RetrievalResult.
        """
        lines: List[str] = []
        lines.append(f"## Graph Knowledge Context [Intent: {result.intent.value}]")

        # 1. Starting Entities
        if result.starting_entities:
            lines.append("### Starting Entities")
            for se in result.starting_entities:
                lines.append(f"- **{se.label}**: `{se.name}` (`{se.id}`) {_format_evidence_tag(se)}")
                self._append_key_properties(lines, se)

        # 2. Target Entities (if multihop path)
        if result.target_entities:
            lines.append("### Target Entities")
            for te in result.target_entities:
                lines.append(f"- **{te.label}**: `{te.name}` (`{te.id}`) {_format_evidence_tag(te)}")

        # 3. Intent-Specific Sections
        if result.intent == QueryIntent.IMPACT_ANALYSIS:
            self._build_impact_section(lines, result)

        elif result.paths:
            lines.append("### Traversed Paths")
            for p in result.paths:
                lines.append(f"- **{p.path_id}** ({p.hop_length} hops): {p.summary}")

        # 4. Connected Graph Entities (deterministically sorted)
        lines.append("### Connected Entities & Relationships")
        sorted_nodes = sorted(
            result.nodes.values(),
            key=lambda n: (
                n.hop_distance,
                LABEL_ORDER.get(n.label, 99),
                n.name.lower(),
                n.id,
            ),
        )

        for n in sorted_nodes:
            if n in result.starting_entities:
                continue  # Already detailed above
            hop_prefix = f"[Hop {n.hop_distance}] " if n.hop_distance > 0 else ""
            lines.append(f"- {hop_prefix}**{n.label}**: `{n.name}` (`{n.id}`) {_format_evidence_tag(n)}")
            self._append_key_properties(lines, n)

        # 5. Explicit Relationships
        if result.relationships:
            lines.append("### Explicit Relationships")
            for r in result.relationships:
                lines.append(f"- `({r.source_id})` -[:{r.rel_type}]-> `({r.target_id})`")

        # 6. Ambiguities & Unresolved References
        if result.ambiguities:
            lines.append("### Ambiguous Matches (Manual Disambiguation Required)")
            for amb in result.ambiguities:
                lines.append(f"- Mention: `{amb.get('mention')}`")
                lines.append(f"  Note: {amb.get('message')}")

        if result.unresolved_references:
            lines.append("### Unresolved References")
            for unres in result.unresolved_references:
                lines.append(f"- Unresolved mention: `{unres}` (no matching node in knowledge graph)")

        # 7. Warnings & Provenance Summary
        if result.warnings:
            lines.append("### Warnings & Limitations")
            for w in result.warnings:
                lines.append(f"- {w}")

        if result.provenance_summary:
            lines.append("### Provenance Summary")
            for p_type, count in sorted(result.provenance_summary.items()):
                lines.append(f"- {p_type}: {count} elements")

        full_context = "\n".join(lines)

        # Enforce character budget deterministically
        if len(full_context) > self.max_characters:
            truncated_marker = "\n\n... [Context truncated to stay within character budget] ..."
            return full_context[: self.max_characters - len(truncated_marker)] + truncated_marker

        return full_context

    def _build_impact_section(self, lines: List[str], result: RetrievalResult) -> None:
        """Builds structured impact analysis separating direct vs indirect impacts."""
        lines.append("### Impact Analysis (Blast Radius)")
        direct_nodes = [n for n in result.nodes.values() if n.hop_distance == 1]
        indirect_nodes = [n for n in result.nodes.values() if n.hop_distance > 1]

        lines.append("#### Direct Impact (Hop 1)")
        if direct_nodes:
            for dn in sorted(direct_nodes, key=lambda x: (LABEL_ORDER.get(x.label, 99), x.name)):
                lines.append(f"- **{dn.label}**: `{dn.name}` (`{dn.id}`) {_format_evidence_tag(dn)}")
        else:
            lines.append("- No direct callers or dependents found.")

        lines.append("#### Indirect Impact (Hop > 1)")
        if indirect_nodes:
            for in_node in sorted(indirect_nodes, key=lambda x: (x.hop_distance, LABEL_ORDER.get(x.label, 99), x.name)):
                lines.append(f"- [Hop {in_node.hop_distance}] **{in_node.label}**: `{in_node.name}` (`{in_node.id}`) {_format_evidence_tag(in_node)}")
        else:
            lines.append("- No indirect transitive dependents found.")

    def _append_key_properties(self, lines: List[str], node: RetrievedNode) -> None:
        """Appends relevant domain properties (HTTP routes, policies, descriptions)."""
        props = node.properties
        notable = []
        if props.get("route"):
            notable.append(f"Route: `{props.get('http_method', 'ANY')} {props['route']}`")
        if props.get("rule_id"):
            notable.append(f"Rule ID: `{props['rule_id']}`")
        if props.get("description"):
            desc = str(props['description'])[:120]
            notable.append(f"Desc: \"{desc}\"")
        if props.get("file_path"):
            notable.append(f"File: `{props['file_path']}`")
        if props.get("workflow_nodes"):
            names = ", ".join(f"`{name}`" for name in props["workflow_nodes"])
            notable.append(
                f"LangGraph workflow nodes ({props.get('workflow_node_count', len(props['workflow_nodes']))}): {names}"
            )

        if props.get("is_api_handler") and props.get("source_excerpt"):
            excerpt = str(props["source_excerpt"]).strip()
            notable.append(f"API handler source:\n```python\n{excerpt}\n```")

        for item in notable:
            lines.append(f"  * {item}")
