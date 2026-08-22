"""Vector store and embedding management."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
from src.excel_loader import AUDIT_KEYWORDS
from src.pdf_loader import DocumentChunk

# SentenceTransformer is imported lazily (inside get_embedding_model) to avoid
# loading ONNX runtime at Streamlit startup, which causes a segfault on
# Python 3.13 / Apple Silicon when joblib's loky backend is also initialised.
_MODEL_CACHE: dict[str, Any] = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # lazy import
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class VectorStore:
    def __init__(self, persist_dir: Path = CHROMA_DIR, model_name: str = EMBEDDING_MODEL):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model = get_embedding_model(model_name)  # lazy — loads on first call
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "page": chunk.page,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "file_name": chunk.file_name,
                "sheet_name": chunk.sheet_name,
                "report_period": chunk.report_period,
            }
            for chunk in chunks
        ]
        embeddings = self._embed(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]
        query_lower = query.lower()
        audit_query = any(word in query_lower for word in AUDIT_KEYWORDS)
        fetch_k = min(max(top_k * 8, 24) if audit_query else max(top_k * 3, top_k), self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        query_terms = {
            word.lower()
            for word in re.findall(r"[A-Za-z]{4,}", query)
        }

        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity = 1 - distance
            section = meta.get("section", "General")
            source_type = meta.get("source_type", "proposal")
            if section not in {"General", "Introduction"}:
                similarity += 0.03

            doc_lower = doc.lower()
            keyword_overlap = sum(1 for term in query_terms if term in doc_lower)
            similarity += min(keyword_overlap * 0.02, 0.12)

            if audit_query and source_type == "audit_data":
                similarity += 0.12
            if audit_query and section == "Audit KPI Summary":
                similarity += 0.1
            sheet_name = (meta.get("sheet_name") or "").lower()
            if "creditor" in query_lower and sheet_name == "creditors analysis":
                similarity += 0.3
            if "debtor" in query_lower and sheet_name == "debtors analysis":
                similarity += 0.3
            if "creditor" in query_lower and section.lower() == "audit kpi summary":
                similarity += 0.2
            if "creditor" in query_lower and (
                "creditors analysis" in doc_lower or "top creditors" in doc_lower
            ):
                similarity += 0.25
            if "debtor" in query_lower and "debtors analysis" in doc_lower:
                similarity += 0.25
            if "creditor" in query_lower and "credit sales" in doc_lower:
                similarity -= 0.15
            if "margin" in query_lower and "margin" in section.lower():
                similarity += 0.15
            if "sales" in query_lower and section in {"Sales Analysis", "Audit KPI Summary"}:
                similarity += 0.1
            if "may" in query_lower and meta.get("report_period", "").lower().startswith("may"):
                similarity += 0.08
            if "april" in query_lower and meta.get("report_period", "").lower().startswith("april"):
                similarity += 0.08
            if "objective" in query_lower and "research objectives" in doc_lower:
                similarity += 0.15
            if "objective" in query_lower and section.lower().startswith("research objectives"):
                similarity += 0.2
            if "tool" in query_lower and "tool/technology" in doc_lower:
                similarity += 0.1
            if meta.get("page") == 0 and source_type == "proposal":
                similarity += 0.05

            hits.append(
                {
                    "text": doc,
                    "page": meta.get("page", "?"),
                    "section": section,
                    "chunk_id": meta.get("chunk_id", ""),
                    "source_type": source_type,
                    "file_name": meta.get("file_name", ""),
                    "sheet_name": meta.get("sheet_name", ""),
                    "report_period": meta.get("report_period", ""),
                    "score": round(similarity, 4),
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:top_k]
