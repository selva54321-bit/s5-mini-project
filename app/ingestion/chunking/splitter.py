from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Recursively splits text into chunks using LangChain's RecursiveCharacterTextSplitter.
    Splits hierarchically on ["\n\n", "\n", " ", ""] to preserve semantic structure.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text), chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        if not text or not text.strip():
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = text_splitter.split_text(text)
        valid_chunks = [c.strip() for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks via RecursiveCharacterTextSplitter")
        return valid_chunks