"""
Grounded Answer Generation Service.
Coordinates graph-only retrieval, evidence formatting, ambiguity safeguards,
and LLM Gateway synthesis with strict anti-hallucination controls.
"""

import time
import logging
import os
from typing import Optional, Dict, Any, List

from app.config import settings

# Logfire tracing is optional in local CLI use.  Suppress only the framework's
# no-configuration warning; enabled deployments retain their normal setup.
if not settings.LOGFIRE_ENABLED:
    os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import logfire
from app.gateway.client import get_langchain_llm, extract_cache_status
from app.graph.retrieval.retriever import GraphRetriever
from app.graph.retrieval.context_builder import GraphContextBuilder
from app.graph.retrieval.subgraph import RetrievalResult, RetrievedNode, RetrievedEdge
from app.graph.retrieval.access_control import AccessControlContext
from app.graph.answering.models import (
    AnswerStatus,
    GroundedAnswerResponse,
    EvidenceSummary,
    EvidenceNodeSummary,
    EvidenceRelationshipSummary,
    ModelMetadata,
)
from app.graph.answering.prompts import build_answering_prompt

logger = logging.getLogger("graph.answering.generator")


class GraphAnswerGenerator:
    """
    End-to-end question answering engine grounded exclusively on deterministic graph evidence.
    """

    def __init__(
        self,
        retriever: Optional[GraphRetriever] = None,
        context_builder: Optional[GraphContextBuilder] = None,
        llm: Optional[Any] = None,
    ):
        self.retriever = retriever or GraphRetriever()
        self.context_builder = context_builder or GraphContextBuilder(max_characters=15000)
        self._llm = llm

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        return get_langchain_llm(feature="graph_answering")

    def generate_answer(
        self,
        question: str,
        context: Optional[AccessControlContext] = None,
        scope_project_urn: Optional[str] = None,
        max_context_chars: int = 15000,
    ) -> GroundedAnswerResponse:
        """
        Executes graph retrieval, verifies evidence sufficiency, and synthesizes a grounded answer.
        """
        if not question or not question.strip():
            return GroundedAnswerResponse(
                question=question or "",
                answer="No question was provided.",
                status=AnswerStatus.ERROR,
                retrieval_intent="none",
                warnings=["Empty question string provided."],
            )

        q_clean = question.strip()
        auth_ctx = context or AccessControlContext()

        with logfire.span("🧠 Graph Grounded Answer Generation", question=q_clean):
            # 1. Graph Retrieval Stage
            with logfire.span("🔍 Graph Retrieval"):
                start_retrieval_time = time.time()
                try:
                    query_or_spec = q_clean
                    if scope_project_urn:
                        query_or_spec = self.retriever.interpreter.interpret(
                            q_clean,
                            scope_project_urn=scope_project_urn,
                        )
                    retrieval_result: RetrievalResult = self.retriever.retrieve(query_or_spec=query_or_spec, context=auth_ctx)
                except PermissionError as pe:
                    logfire.error(f"Access Control Error: {pe}")
                    return GroundedAnswerResponse(
                        question=q_clean,
                        answer=f"Access denied: {pe}",
                        status=AnswerStatus.ERROR,
                        retrieval_intent="unauthorized",
                        warnings=[str(pe)],
                    )
                except Exception as ex:
                    logfire.error(f"Graph Retrieval Error: {ex}")
                    return GroundedAnswerResponse(
                        question=q_clean,
                        answer=f"Graph retrieval failed due to an error: {ex}",
                        status=AnswerStatus.ERROR,
                        retrieval_intent="unknown",
                        warnings=[str(ex)],
                    )

            retrieval_duration = time.time() - start_retrieval_time
            logfire.info(
                "Graph retrieval completed",
                nodes_retrieved=len(retrieval_result.nodes),
                edges_retrieved=len(retrieval_result.relationships),
                duration_seconds=round(retrieval_duration, 3),
            )

            # Build structured evidence summary
            evidence_summary = self._build_evidence_summary(retrieval_result)
            resolved_entity_names = [n.name for n in retrieval_result.starting_entities]

            # 2. Ambiguity Safeguard: Return ambiguous status directly without calling LLM
            if retrieval_result.ambiguities:
                amb_notes = [f"- Mention: '{a.get('mention')}' -> {a.get('message')}" for a in retrieval_result.ambiguities]
                amb_text = (
                    "The question contains ambiguous entity references that match multiple candidates in the knowledge graph:\n\n"
                    + "\n".join(amb_notes)
                    + "\n\nPlease provide a fully-qualified name, repository, or file path to disambiguate."
                )
                return GroundedAnswerResponse(
                    question=q_clean,
                    answer=amb_text,
                    status=AnswerStatus.AMBIGUOUS,
                    retrieval_intent=retrieval_result.intent.value,
                    entities_resolved=resolved_entity_names,
                    evidence=evidence_summary,
                    warnings=retrieval_result.warnings,
                )

            # 3. Insufficient Evidence Safeguard: Empty or missing graph nodes
            if not retrieval_result.nodes:
                missing_refs = (
                    f" Unresolved references: {', '.join(retrieval_result.unresolved_references)}."
                    if retrieval_result.unresolved_references
                    else ""
                )
                insufficient_msg = (
                    f"Insufficient graph evidence: No matching entities or relationships were found in the knowledge graph "
                    f"to answer the question.{missing_refs}"
                )
                return GroundedAnswerResponse(
                    question=q_clean,
                    answer=insufficient_msg,
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    retrieval_intent=retrieval_result.intent.value,
                    entities_resolved=resolved_entity_names,
                    evidence=evidence_summary,
                    warnings=retrieval_result.warnings,
                )

            # 4. Context Construction Stage
            with logfire.span("📝 Graph Context Construction"):
                context_str = self.context_builder.build_context(retrieval_result)
                if len(context_str) > max_context_chars:
                    context_str = context_str[:max_context_chars] + "\n... [Context truncated to size limit] ..."

            prompt = build_answering_prompt(question=q_clean, graph_context=context_str)

            # 5. LLM Gateway Synthesis Stage
            with logfire.span("🤖 LLM Gateway Invocation"):
                t0 = time.time()
                try:
                    llm = self._get_llm()
                    response = llm.invoke(prompt)
                    latency_ms = (time.time() - t0) * 1000.0

                    answer_content = response.content if hasattr(response, "content") else str(response)
                    if isinstance(answer_content, list):
                        answer_content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in answer_content])

                    if not answer_content or not str(answer_content).strip():
                        return GroundedAnswerResponse(
                            question=q_clean,
                            answer="The model returned an empty response.",
                            status=AnswerStatus.ERROR,
                            retrieval_intent=retrieval_result.intent.value,
                            entities_resolved=resolved_entity_names,
                            evidence=evidence_summary,
                            warnings=retrieval_result.warnings + ["Empty response received from LLM Gateway."],
                        )

                    cache_status = extract_cache_status(response)
                    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or settings.GEMINI_MODEL

                    model_meta = ModelMetadata(
                        model_name=str(model_name),
                        temperature=getattr(llm, "temperature", 0.0),
                        provider="google_genai",
                        cache_status=cache_status,
                        latency_ms=round(latency_ms, 2),
                    )

                    logfire.info(
                        "Answer synthesized successfully",
                        latency_ms=round(latency_ms, 2),
                        cache_status=cache_status,
                    )

                    return GroundedAnswerResponse(
                        question=q_clean,
                        answer=str(answer_content).strip(),
                        status=AnswerStatus.ANSWERED,
                        retrieval_intent=retrieval_result.intent.value,
                        entities_resolved=resolved_entity_names,
                        evidence=evidence_summary,
                        warnings=retrieval_result.warnings,
                        model_metadata=model_meta,
                    )

                except Exception as gateway_err:
                    latency_ms = (time.time() - t0) * 1000.0
                    logfire.error(f"LLM Gateway invocation failed: {gateway_err}")
                    return GroundedAnswerResponse(
                        question=q_clean,
                        answer=f"Answer synthesis failed due to an LLM Gateway error: {gateway_err}",
                        status=AnswerStatus.ERROR,
                        retrieval_intent=retrieval_result.intent.value,
                        entities_resolved=resolved_entity_names,
                        evidence=evidence_summary,
                        warnings=retrieval_result.warnings + [f"LLM Gateway error: {gateway_err}"],
                        model_metadata=ModelMetadata(
                            model_name=settings.GEMINI_MODEL,
                            temperature=0.0,
                            provider="google_genai",
                            latency_ms=round(latency_ms, 2),
                        ),
                    )

    def _build_evidence_summary(self, result: RetrievalResult) -> EvidenceSummary:
        """Constructs serializable evidence summary from retrieved subgraph."""
        node_summaries = []
        for n in result.nodes.values():
            loc = n.provenance_file
            if loc and n.provenance_lines:
                loc += f":{n.provenance_lines}"
            node_summaries.append(
                EvidenceNodeSummary(
                    id=n.id,
                    name=n.name,
                    label=n.label,
                    source_location=loc,
                    provenance_type=n.provenance_type,
                    confidence=n.confidence,
                    properties=n.properties,
                )
            )

        rel_summaries = []
        for r in result.relationships:
            rel_summaries.append(
                EvidenceRelationshipSummary(
                    source_id=r.source_id,
                    target_id=r.target_id,
                    rel_type=r.rel_type,
                    direction=r.direction,
                    hop=r.hop,
                    provenance_type=r.provenance_type,
                    confidence=r.confidence,
                )
            )

        path_summaries = [p.summary for p in result.paths]

        return EvidenceSummary(
            nodes=node_summaries,
            relationships=rel_summaries,
            paths=path_summaries,
            provenance_counts=dict(result.provenance_summary),
        )
