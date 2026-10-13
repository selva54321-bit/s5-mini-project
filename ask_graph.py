"""
Interactive Graph-RAG Query CLI.
Ask natural-language questions about any ingested repository or document.
Answers are grounded exclusively on deterministic graph traversal evidence.

Usage:
    python ask_graph.py                          # Interactive mode
    python ask_graph.py "What functions exist?"  # Single question mode
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.connection import db_manager
from app.graph.answering.generator import GraphAnswerGenerator
from app.graph.answering.models import AnswerStatus


def print_answer(response):
    """Pretty-print a GroundedAnswerResponse."""
    status_icons = {
        AnswerStatus.ANSWERED: "✅",
        AnswerStatus.INSUFFICIENT_EVIDENCE: "⚠️",
        AnswerStatus.AMBIGUOUS: "🔀",
        AnswerStatus.ERROR: "❌",
    }
    icon = status_icons.get(response.status, "❓")

    print(f"\n{icon} Status: {response.status.value.upper()}")
    print(f"   Intent: {response.retrieval_intent}")

    if response.entities_resolved:
        print(f"   Entities Found: {', '.join(response.entities_resolved)}")

    # Show evidence
    if response.evidence and response.evidence.nodes:
        print(f"\n   📊 Evidence: {len(response.evidence.nodes)} nodes, "
              f"{len(response.evidence.relationships)} relationships")
        for n in response.evidence.nodes[:10]:  # Show first 10
            print(f"      • [{n.label}] {n.name}")
        if len(response.evidence.nodes) > 10:
            print(f"      ... and {len(response.evidence.nodes) - 10} more")

    if response.evidence and response.evidence.relationships:
        for r in response.evidence.relationships[:5]:
            print(f"      → {r.source_id} -[{r.rel_type}]-> {r.target_id}")
        if len(response.evidence.relationships) > 5:
            print(f"      ... and {len(response.evidence.relationships) - 5} more")

    # Show answer
    print(f"\n   💬 Answer:")
    for line in response.answer.split("\n"):
        print(f"      {line}")

    if response.warnings:
        print(f"\n   ⚠️  Warnings:")
        for w in response.warnings:
            print(f"      - {w}")

    if response.model_metadata:
        print(f"\n   🤖 Model: {response.model_metadata.model_name} | "
              f"Latency: {response.model_metadata.latency_ms}ms")


def main():
    print("=" * 60)
    print("  🧠 Graph-RAG Interactive Query System")
    print("=" * 60)

    # Check connection
    health = db_manager.health_check()
    is_healthy = health.get("status") == "HEALTHY"

    if is_healthy:
        print("✅ Neo4j: CONNECTED (querying live knowledge graph)")
    else:
        print("❌ Neo4j: NOT CONNECTED")
        print("   Fix your .env configuration and try again.")
        return

    generator = GraphAnswerGenerator()

    # Single question mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n🎯 Question: {question}")
        print("-" * 60)
        response = generator.generate_answer(question)
        print_answer(response)
        return

    # Interactive mode
    print("\nType your questions below. Type 'quit' or 'exit' to stop.")
    print("Example questions about your ingested repos:")
    print("  • What files are in the eqrev repository?")
    print("  • What functions exist in the project?")
    print("  • What API endpoints are defined?")
    print("  • What packages does the project depend on?")
    print("  • Show me the structure of the codebase")
    print("-" * 60)

    while True:
        try:
            question = input("\n🔍 Ask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        print("-" * 60)
        try:
            response = generator.generate_answer(question)
            print_answer(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()
