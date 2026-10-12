# Enterprise Agentic RAG Pipeline

An enterprise-grade, agentic Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **Gemini Embeddings**, **Groq (Llama 3.3 70B)**, **Qdrant Vector Database**, **FlashRank Reranker**, and **Logfire Observability**.

---

## 🏗 System Architecture & Workflow

```mermaid
flowchart TD
    User([User Query]) --> Planner[Planner Node]
    Planner -->|Conversational / Direct| Responder[Responder Node]
    Planner -->|Needs Context Search| Retriever[Retriever Node]
    Retriever --> Qdrant[(Qdrant Vector DB)]
    Retriever --> FlashRank[FlashRank Cross-Encoder Reranker]
    FlashRank --> Responder
    Responder --> Groq[Groq Llama-3.3-70b Engine]
    Groq --> Output([Final Answer / Logfire Tracing])
```

---

## ✨ Features

- **Agentic Workflow (LangGraph)**: Multi-step execution graph supporting planning, state-aware routing, retrieval, and response synthesis.
- **Multi-Format Ingestion**: Supports parsing for **PDF**, **HTML**, **TXT**, **DOCX**, and **PPTX** documents.
- **Dense Vector Search**: Powered by Google Gemini Embeddings (`text-embedding-004`) stored inside **Qdrant Cloud/Local**.
- **Two-Stage Retrieval & Reranking**: Ultra-fast vector retrieval followed by cross-encoder re-ranking using **FlashRank**.
- **LLM Gateway Integration**: Unified routing with fallback handling powered by **Portkey** and **Groq (Llama 3.3 70B)**.
- **Production Observability & Guardrails**: Full execution tracing via **Pydantic Logfire** & input/output validation via **NVIDIA NeMo Guardrails**.

---

## 📂 Project Structure

```text
├── app/
│   ├── agent/             # LangGraph state machine, planner, retriever, responder nodes
│   │   ├── nodes/         # Individual graph nodes (planner.py, retriever.py, responder.py)
│   │   ├── graph.py       # LangGraph DAG workflow definition
│   │   └── state.py       # Agent state schema
│   ├── gateway/           # LLM Gateway client (Portkey / Groq routing)
│   ├── ingestion/         # Document parsing, chunking, and Qdrant vector indexing
│   │   ├── chunking/      # Text splitters and chunking strategies
│   │   ├── loaders/       # Custom parsers (PDF, HTML, Office, Text)
│   │   └── processor.py   # Full ingestion pipeline processor
│   ├── services/          # Core backend services
│   │   └── retrieval/     # Embedding service, Qdrant client, FlashRank reranking
│   └── config.py          # Centralized configuration & environment settings
├── DATA/                  # Local directory for input raw documents
├── processed_data/        # Processed chunk metadata outputs
├── requirements.txt       # Python dependencies
├── index.html             # Basic frontend interface for document upload
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python `3.10` or higher
- A [Qdrant Cloud](https://qdrant.tech/) account (or local Qdrant instance)
- API Keys for **Google Gemini**, **Groq**, and **Qdrant**

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


### 3. Environment Configuration

Create a `.env` file in the project root directory:

```ini
# Gemini API Key (for embeddings)
GEMINI_API_KEY=your_gemini_api_key

# Qdrant Vector Database
QDRANT_CLUSTER_ENDPOINT=https://your-qdrant-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Groq LLM
GROQ_API_KEY=your_groq_api_key
GROQ_FALLBACK_API_KEY=your_groq_fallback_api_key

# Portkey Gateway (optional). Leave both unset to call Groq directly.
# PORTKEY_CONFIG_ID must reference a Portkey configuration that defines the
# GROQ_SLUG provider/virtual-key alias; GROQ_API_KEY alone does not do that.
# PORTKEY_API_KEY=your_portkey_api_key
# PORTKEY_CONFIG_ID=pc-xxx
# GROQ_SLUG=your_portkey_groq_alias

# Observability (Optional)
LOGFIRE_TOKEN=your_logfire_token
```

---

## 🛠 Usage

### Ingesting Documents

To process, chunk, embed, and index documents into Qdrant, place your files inside the `DATA/` directory or trigger the ingestion processor:

```python
from app.ingestion.processor import process_file

process_file(
    file_path="DATA/sample_document.pdf",
    filename="sample_document.pdf",
    source_type="pdf"
)
```


### Executing the RAG Agent

You can invoke the compiled agentic graph in Python:

```python
from app.agent.graph import rag_agent

config = {"configurable": {"thread_id": "session-1"}}

response = rag_agent.invoke(
    {"messages": [("user", "What are the key findings in the Q3 financial report?")]},
    config=config
)

print(response["messages"][-1].content)
```

---

## 🧪 Evaluation & Tracing

- **LangChain / LangSmith / Logfire**: Automated tracing and observability are integrated into the execution flow.
- **RAGAS & DeepEval**: Evaluates context recall, precision, and response faithfulness using standard evaluation frameworks.

---

## 📄 License

This project is licensed under the MIT License.
