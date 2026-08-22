"""RAG engine: retrieval + answer generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

from config import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PDF_PATH,
    PROCESSED_DIR,
    SUMMARY_PATH,
    TOP_K,
)
from src.excel_loader import AUDIT_KEYWORDS, load_excel_chunks
from src.pdf_loader import DocumentChunk, chunk_text, extract_pages
from src.vector_store import VectorStore


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    backend: str


SYSTEM_PROMPT = """You are a research and financial intelligence assistant for an MSc thesis on
SME restaurant financial risk assessment in Kathmandu.

You have access to:
1) The research proposal (methodology, objectives, framework design)
2) Real audit data from Emilio's Pizza (sales, P&L, cash flow, creditors, debtors, inventory, wastage)

Answer ONLY using the provided context. Distinguish between proposal/theory and actual audit figures.
Cite source type (proposal page or audit file/sheet/period) when referencing specific content.
If the context does not contain enough information, say so clearly.
Be concise, academic, and accurate."""


class RAGEngine:
    def __init__(self):
        self.store = VectorStore()

    def is_indexed(self) -> bool:
        return self.store.count() > 0

    def ingest(self, pdf_path=PDF_PATH, force: bool = False) -> dict[str, Any]:
        if force:
            self.store.reset()

        if self.store.count() > 0 and not force:
            return {
                "status": "skipped",
                "chunks": self.store.count(),
                "message": "Index already exists. Use force=True to rebuild.",
            }

        pages = extract_pages(pdf_path)
        chunks = chunk_text(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        if SUMMARY_PATH.exists():
            summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
            chunks.append(
                DocumentChunk(
                    text=summary_text,
                    page=0,
                    chunk_id="summary_0001",
                    section="Research Summary",
                )
            )
            for index, section in enumerate(
                [s.strip() for s in summary_text.split("\n## ") if s.strip()],
                start=1,
            ):
                if not section.startswith("#"):
                    section = "## " + section
                chunks.append(
                    DocumentChunk(
                        text=section,
                        page=0,
                        chunk_id=f"summary_{index+1:04d}",
                        section=section.split("\n", 1)[0].replace("#", "").strip(),
                    )
                )

        audit_chunks = load_excel_chunks(DATA_DIR)
        chunks.extend(audit_chunks)

        added = self.store.add_chunks(chunks)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        full_text_path = PROCESSED_DIR / "extracted_text.txt"
        with full_text_path.open("w", encoding="utf-8") as handle:
            for page_num, text in pages:
                handle.write(f"\n--- Page {page_num} ---\n{text}\n")

        audit_manifest = PROCESSED_DIR / "audit_manifest.txt"
        with audit_manifest.open("w", encoding="utf-8") as handle:
            handle.write(f"Audit chunks indexed: {len(audit_chunks)}\n")
            for chunk in audit_chunks:
                if chunk.section == "Audit Data Overview":
                    handle.write(
                        f"- {chunk.file_name} | {chunk.report_period} | {chunk.text.splitlines()[-1]}\n"
                    )

        return {
            "status": "success",
            "pages": len(pages),
            "proposal_chunks": len(chunks) - len(audit_chunks),
            "audit_chunks": len(audit_chunks),
            "chunks": added,
            "text_path": str(full_text_path),
            "audit_manifest": str(audit_manifest),
        }

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if not self.is_indexed():
            self.ingest()
        return self.store.search(question, top_k=top_k)

    def _format_source_label(self, hit: dict[str, Any]) -> str:
        if hit.get("source_type") == "audit_data":
            parts = [
                "Audit Data",
                hit.get("report_period") or "Unknown period",
                hit.get("file_name") or "",
                hit.get("sheet_name") or hit.get("section") or "",
            ]
            return " | ".join(p for p in parts if p)
        return f"Proposal p.{hit.get('page', '?')} | {hit.get('section', 'General')}"

    def _build_context(self, hits: list[dict[str, Any]]) -> str:
        blocks = []
        for index, hit in enumerate(hits, start=1):
            label = self._format_source_label(hit)
            blocks.append(
                f"[Source {index} | {label} | Relevance: {hit['score']}]\n{hit['text']}"
            )
        return "\n\n".join(blocks)

    def _generate_openai(self, question: str, context: str) -> str | None:
        if not OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def _generate_ollama(self, question: str, context: str) -> str | None:
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("message", {}).get("content", "").strip() or None
        except Exception:
            return None

    def _is_useful_sentence(self, sentence: str) -> bool:
        if len(sentence) < 40:
            return False
        if re.search(r"\.{5,}", sentence):
            return False
        if re.fullmatch(r"[\d.\s]+", sentence):
            return False
        if re.fullmatch(r"#+\s*[\w\s]+", sentence.strip()):
            return False
        lowered = sentence.lower()
        noise_terms = ("table of contents", "list of figures", "accessed:")
        return not any(term in lowered for term in noise_terms)

    def _generate_extractive(self, question: str, hits: list[dict[str, Any]]) -> str:
        """Fallback: synthesize answer from top retrieved chunks without an LLM."""
        if not hits:
            return "No relevant content found in the research proposal or audit data index."

        priority = next(
            (
                hit
                for hit in hits
                if hit.get("section") in {"Audit KPI Summary", "Creditors Analysis", "Debtors Analysis"}
                or hit.get("sheet_name") in {"Creditors Analysis", "Debtors Analysis", "KPI Summary"}
            ),
            None,
        )
        if priority and priority.get("source_type") == "audit_data":
            label = self._format_source_label(priority)
            return (
                f"Based on audit data ({label}):\n\n"
                f"{priority['text'][:2200]}{'...' if len(priority['text']) > 2200 else ''}"
            )

        keywords = {
            word.lower()
            for word in re.findall(r"[A-Za-z]{4,}", question)
            if word.lower() not in {"what", "which", "where", "when", "does", "this", "that", "with", "from", "about"}
        }

        ranked_sentences: list[tuple[float, str, dict[str, Any]]] = []
        for hit in hits:
            sentences = re.split(r"(?<=[.!?])\s+", hit["text"])
            for sentence in sentences:
                sentence = sentence.strip()
                if not self._is_useful_sentence(sentence):
                    continue
                lower = sentence.lower()
                overlap = sum(1 for kw in keywords if kw in lower)
                score = overlap + hit["score"]
                ranked_sentences.append((score, sentence, hit))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen = set()
        for _, sentence, hit in ranked_sentences[:6]:
            key = sentence[:80]
            if key in seen:
                continue
            seen.add(key)
            selected.append(f"{sentence} ({self._format_source_label(hit)})")

        if not selected:
            top = hits[0]
            return (
                f"Based on the most relevant section ({top['section']}, p. {top['page']}):\n\n"
                f"{top['text'][:900]}{'...' if len(top['text']) > 900 else ''}"
            )

        intro = (
            "Based on retrieved sections of the research proposal and audit data, "
            "here is a synthesized answer (extractive mode — no external LLM used):"
        )
        return intro + "\n\n" + "\n\n".join(f"- {line}" for line in selected)

    def ask(self, question: str, top_k: int = TOP_K) -> RAGResponse:
        query_lower = question.lower()
        audit_query = any(word in query_lower for word in AUDIT_KEYWORDS)
        effective_top_k = max(top_k, 8) if audit_query else top_k
        hits = self.retrieve(question, top_k=effective_top_k)
        context = self._build_context(hits[:top_k])

        answer = self._generate_openai(question, context)
        backend = "openai"
        if not answer:
            answer = self._generate_ollama(question, context)
            backend = "ollama"
        if not answer:
            answer = self._generate_extractive(question, hits)
            backend = "extractive"

        return RAGResponse(
            question=question,
            answer=answer,
            sources=hits[:top_k],
            backend=backend,
        )
