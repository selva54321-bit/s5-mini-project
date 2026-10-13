"""
Interactive / End-to-End Test Runner for Graph-RAG Answering Pipeline.
Demonstrates how real questions are answered using deterministic multi-hop graph retrieval
and LLM synthesis with verifiable provenance citations.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.connection import db_manager
from app.graph.answering.generator import GraphAnswerGenerator
from app.graph.answering.models import AnswerStatus


def run_graph_query(query: str):
    print("\n" + "=" * 70)
    print(f"🎯 Query: {query}")
    print("=" * 70)

    # 1. Check live Neo4j status
    health = db_manager.health_check()
    is_healthy = health.get("status") == "HEALTHY"
    if is_healthy:
        print("✅ Neo4j Connection: ONLINE (Querying live knowledge graph)")
        generator = GraphAnswerGenerator()
    else:
        print("ℹ️  Neo4j Connection: OFFLINE / Not Configured")
        print("   (Using deterministic graph knowledge fixture to demonstrate flow)\n")
        from tests.test_graph_retrieval import build_graph_fixture
        from app.graph.retrieval.retriever import GraphRetriever
        from app.graph.retrieval.context_builder import GraphContextBuilder

        fixture_store = build_graph_fixture()
        retriever = GraphRetriever(store=fixture_store)
        context_builder = GraphContextBuilder()
        generator = GraphAnswerGenerator(retriever=retriever, context_builder=context_builder)

    # 2. Generate grounded answer
    response = generator.generate_answer(query)

    # 3. Display structured results
    print(f"\n[Status]:           {response.status.value.upper()}")
    print(f"[Retrieval Intent]: {response.retrieval_intent}")
    print(f"[Entities Found]:   {response.entities_resolved or 'None'}")

    print("\n--- 🔍 Retrieved Graph Evidence ---")
    if response.evidence.nodes:
        print(f"  Nodes ({len(response.evidence.nodes)}):")
        for n in response.evidence.nodes:
            loc = f" [{n.source_location}]" if n.source_location else ""
            prov = f" ({n.provenance_type})" if n.provenance_type else ""
            print(f"    * {n.label}: {n.name} (URN: {n.id}){loc}{prov}")
    else:
        print("  Nodes: None found.")

    if response.evidence.relationships:
        print(f"  Relationships ({len(response.evidence.relationships)}):")
        for r in response.evidence.relationships:
            print(f"    * ({r.source_id}) -[:{r.rel_type}]-> ({r.target_id}) [Hop {r.hop}]")

    if response.warnings:
        print("\n--- ⚠️ Warnings & Safeguards ---")
        for w in response.warnings:
            print(f"  * {w}")

    print("\n--- 📝 Grounded Answer from LLM ---")
    print(response.answer)

    if response.model_metadata:
        print(f"\n[Model]: {response.model_metadata.model_name} | Latency: {response.model_metadata.latency_ms}ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test 1: Direct Entity Lookup (Real Class node in Neo4j)
    run_graph_query("Tell me about class AgentState")

    # Test 2: Caller Lookup (Real Function with incoming CALLS in Neo4j)
    run_graph_query("Who calls verify_connectivity?")

    # Test 3: Callee Lookup (Real Function with outgoing CALLS in Neo4j)
    run_graph_query("What functions does generate_answer call?")

    # Test 4: Document / Policy Lookup (Real Document & Policy nodes in Neo4j)
    run_graph_query("What does architecture.pptx document?")

    # Test 5: Insufficient Evidence Handling (Genuinely nonexistent entity)
    run_graph_query("Who calls auth.security.authenticate?")
