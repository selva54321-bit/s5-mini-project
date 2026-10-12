"""
Document Discovery and Parser Module (Task 1).
Supports multi-format parsing (.pdf, .docx, .pptx, .html, .md, .txt) reusing existing loaders.
Preserves stable document URNs, titles, page/slide references, and structured sections without macro execution.
"""

import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.graph.ingestion.docs.identifiers import build_doc_urn

SUPPORTED_DOC_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".md",
    ".txt",
}


@dataclass
class DocSection:
    section_index: int
    page_or_slide: Optional[int]
    heading: str
    content: str


@dataclass
class ParsedDocument:
    urn: str
    title: str
    rel_path: str
    abs_path: str
    source_type: str
    file_size_bytes: int
    content_hash: str
    full_text: str
    sections: List[DocSection] = field(default_factory=list)
    total_pages: int = 1
    success: bool = True
    error_message: Optional[str] = None


def compute_file_hash(abs_path: str) -> str:
    """Computes SHA-256 hash of file contents for deduplication."""
    sha = hashlib.sha256()
    with open(abs_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def discover_documents(dir_path: str) -> List[str]:
    """Scans directory for supported document files, sorting deterministically."""
    root = Path(dir_path).resolve()
    if not root.exists() or not root.is_dir():
        return []

    discovered = []
    for root_dir, _, files in os.walk(str(root)):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_DOC_EXTENSIONS:
                abs_f = os.path.join(root_dir, f)
                discovered.append(abs_f)

    discovered.sort()
    return discovered


def parse_document(abs_path: str, rel_path: str, project_urn: str) -> ParsedDocument:
    """
    Parses a single document safely into structured sections and metadata.
    Reuses project loaders where available and handles corrupt/empty files gracefully.
    """
    doc_urn = build_doc_urn(project_urn, rel_path)
    ext = os.path.splitext(abs_path)[1].lower()
    source_type = ext.lstrip(".")

    try:
        file_size = os.path.getsize(abs_path)
    except Exception as e:
        return ParsedDocument(
            urn=doc_urn,
            title=os.path.basename(rel_path),
            rel_path=rel_path,
            abs_path=abs_path,
            source_type=source_type,
            file_size_bytes=0,
            content_hash="",
            full_text="",
            success=False,
            error_message=f"Could not access file: {e}",
        )

    if file_size == 0:
        return ParsedDocument(
            urn=doc_urn,
            title=os.path.basename(rel_path),
            rel_path=rel_path,
            abs_path=abs_path,
            source_type=source_type,
            file_size_bytes=0,
            content_hash="",
            full_text="",
            success=False,
            error_message="Empty file (0 bytes)",
        )

    content_hash = compute_file_hash(abs_path)
    sections: List[DocSection] = []
    full_text = ""
    total_pages = 1

    try:
        # 1. PDF Documents
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(abs_path)
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append(
                        DocSection(
                            section_index=idx + 1,
                            page_or_slide=idx + 1,
                            heading=f"Page {idx + 1}",
                            content=page_text.strip(),
                        )
                    )
            full_text = "\n\n".join(s.content for s in sections)

        # 2. DOCX / PPTX Office Documents
        elif ext in (".docx", ".pptx"):
            from app.ingestion.loaders.office import parse_office
            full_text = parse_office(abs_path) or ""
            # Segment paragraphs into logical sections
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                # Detect heading style line (short first line)
                first_line = para.split("\n")[0]
                heading = first_line[:60] if len(first_line) < 80 else f"Section {idx + 1}"
                sections.append(
                    DocSection(
                        section_index=idx + 1,
                        page_or_slide=idx + 1,
                        heading=heading,
                        content=para,
                    )
                )

        # 3. HTML Documents
        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()

            # Extract headers and body
            headings = soup.find_all(["h1", "h2", "h3", "h4"])
            if headings:
                for idx, h in enumerate(headings):
                    heading_text = h.get_text().strip()
                    # Capture next siblings until next header
                    content_parts = []
                    curr = h.next_sibling
                    while curr and (not getattr(curr, "name", None) or curr.name not in ["h1", "h2", "h3", "h4"]):
                        text_val = curr.get_text().strip() if hasattr(curr, "get_text") else str(curr).strip()
                        if text_val:
                            content_parts.append(text_val)
                        curr = curr.next_sibling
                    sec_content = "\n".join(content_parts) or heading_text
                    sections.append(
                        DocSection(
                            section_index=idx + 1,
                            page_or_slide=None,
                            heading=heading_text,
                            content=sec_content,
                        )
                    )
            full_text = soup.get_text(separator="\n").strip()

        # 4. Markdown & Plain Text
        else:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read().strip()
            # Split on markdown headings
            chunks = full_text.split("\n#")
            for idx, chunk in enumerate(chunks):
                if chunk.strip():
                    heading = chunk.split("\n")[0].strip("# ")
                    sections.append(
                        DocSection(
                            section_index=idx + 1,
                            page_or_slide=None,
                            heading=heading[:80] or f"Section {idx + 1}",
                            content=chunk.strip(),
                        )
                    )

    except Exception as e:
        return ParsedDocument(
            urn=doc_urn,
            title=os.path.basename(rel_path),
            rel_path=rel_path,
            abs_path=abs_path,
            source_type=source_type,
            file_size_bytes=file_size,
            content_hash=content_hash,
            full_text="",
            success=False,
            error_message=f"Parse failure: {e}",
        )

    if not full_text.strip():
        return ParsedDocument(
            urn=doc_urn,
            title=os.path.basename(rel_path),
            rel_path=rel_path,
            abs_path=abs_path,
            source_type=source_type,
            file_size_bytes=file_size,
            content_hash=content_hash,
            full_text="",
            success=False,
            error_message="No readable text extracted",
        )

    # Derive human-friendly title
    title = os.path.basename(rel_path)
    if sections and sections[0].heading and not sections[0].heading.startswith("Page"):
        title = sections[0].heading

    return ParsedDocument(
        urn=doc_urn,
        title=title,
        rel_path=rel_path,
        abs_path=abs_path,
        source_type=source_type,
        file_size_bytes=file_size,
        content_hash=content_hash,
        full_text=full_text,
        sections=sections,
        total_pages=total_pages,
        success=True,
    )
