"""PDF text extraction and normalization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    text: str
    page: int
    chunk_id: str
    section: str = ""
    source_type: str = "proposal"
    file_name: str = ""
    sheet_name: str = ""
    report_period: str = ""


def normalize_text(text: str) -> str:
    """Collapse PDF extraction artifacts into readable prose."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_section(text: str) -> str:
    """Infer section title from chunk content."""
    section_patterns = [
        (r"\bIntroduction\b", "Introduction"),
        (r"\bMotivation\b", "Motivation"),
        (r"\bCognitive\s+Bias", "Cognitive Bias"),
        (r"\bAim\s+and\s+Objectives\b", "Aim and Objectives"),
        (r"\bResearch\s+Aim\b", "Research Aim"),
        (r"\bResearch\s+Objectives\b", "Research Objectives"),
        (r"\bJustification\b", "Justification"),
        (r"\bResearch\s+Questions?\b", "Research Questions"),
        (r"\bHypotheses\b", "Hypotheses"),
        (r"\bResearch\s+Methodology\b", "Research Methodology"),
        (r"\bLiterature\s+Review\b", "Literature Review"),
        (r"\bDescriptive\s+analytics\b", "Descriptive Analytics"),
        (r"\bDiagnostic\s+Analysis\b", "Diagnostic Analysis"),
        (r"\bPredictive\s+Analytics\b", "Predictive Analytics"),
        (r"\bPrescriptive\s+Analytics\b", "Prescriptive Analytics"),
        (r"\bCase\s+Study\b", "Case Studies"),
        (r"\bIntegration\s+of\s+Tools\b", "Tools and Technologies"),
        (r"\bData\s+Pipeline\b", "Data Pipeline"),
        (r"\bConclusion\b", "Conclusion"),
        (r"\bReferences\b", "References"),
    ]
    for pattern, name in section_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return "General"


def is_low_quality_page(text: str) -> bool:
    """Skip table-of-contents and figure-list pages that harm retrieval."""
    lowered = text.lower()
    if "table of contents" in lowered or "list of figures" in lowered:
        return True
    dot_runs = len(re.findall(r"\.{5,}", text))
    return dot_runs >= 3 and len(text) < 2500


def is_low_quality_chunk(text: str) -> bool:
    """Filter chunks that look like TOC noise."""
    if len(text) < 80:
        return True
    dot_runs = len(re.findall(r"\.{5,}", text))
    if dot_runs >= 2:
        return True
    if text.lower().startswith("table of contents"):
        return True
    return False


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract normalized text from each PDF page."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = normalize_text(raw)
        if cleaned and not is_low_quality_page(cleaned):
            pages.append((index, cleaned))
    return pages


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Split page text into overlapping chunks with metadata."""
    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for page_num, page_text in pages:
        if len(page_text) <= chunk_size:
            if not is_low_quality_chunk(page_text):
                chunk_counter += 1
                chunks.append(
                    DocumentChunk(
                        text=page_text,
                        page=page_num,
                        chunk_id=f"chunk_{chunk_counter:04d}",
                        section=detect_section(page_text),
                    )
                )
            continue

        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            if end < len(page_text):
                boundary = page_text.rfind(". ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary + 1

            piece = page_text[start:end].strip()
            if piece and not is_low_quality_chunk(piece):
                chunk_counter += 1
                chunks.append(
                    DocumentChunk(
                        text=piece,
                        page=page_num,
                        chunk_id=f"chunk_{chunk_counter:04d}",
                        section=detect_section(piece),
                    )
                )

            if end >= len(page_text):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks
