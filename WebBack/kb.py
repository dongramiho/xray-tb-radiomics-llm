"""Knowledge-base retrieval (RAG) for the stage-1 report.

The eight TB documents are split into paragraph chunks, embedded once (OpenAI ``text-embedding-3-small``,
or a deterministic hashing embedder when ``LLM_BACKEND=fake``), cached next to the documents, and the
top-k chunks for a case are injected into the prompt with stable ids such as ``tb_5#2`` so the report's
``supporting_evidence`` can cite them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

KB_FILE_PATTERN = "tb_*.txt"
INDEX_FILENAME = ".kb_index.json"
MAX_CHUNK_CHARS = 700


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    index: int
    text: str


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            response = self._client.embeddings.create(model=self.model, input=texts[start : start + 64])
            vectors.extend(item.embedding for item in response.data)
        return _normalize(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Offline stand-in: character n-gram hashing. Deterministic, no network, adequate for tests."""

    name = "hashing:char-ngrams"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", text.lower())
            for token in tokens:
                for n in (2, 3):
                    for i in range(max(1, len(token) - n + 1)):
                        gram = token[i : i + n]
                        bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.dim
                        matrix[row, bucket] += 1.0
        return _normalize(matrix)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def split_into_chunks(doc_id: str, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split on blank lines, then merge paragraphs until ``max_chars`` so headings stay with their body."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if buffer and len(candidate) > max_chars:
            chunks.append(Chunk(id=f"{doc_id}#{len(chunks)}", doc=doc_id, index=len(chunks), text=buffer))
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(Chunk(id=f"{doc_id}#{len(chunks)}", doc=doc_id, index=len(chunks), text=buffer))
    return chunks


def load_chunks(kb_dir: Path) -> list[Chunk]:
    files = sorted(Path(kb_dir).glob(KB_FILE_PATTERN), key=lambda p: (len(p.stem), p.stem))
    chunks: list[Chunk] = []
    for file in files:
        chunks.extend(split_into_chunks(file.stem, file.read_text(encoding="utf-8")))
    if not chunks:
        raise RuntimeError(f"No knowledge-base documents ({KB_FILE_PATTERN}) found in {kb_dir}")
    return chunks


def _fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    digest = hashlib.sha256(embedder_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()[:16]


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.chunk_ids = {c.id for c in chunks}

    @classmethod
    def build(cls, kb_dir: Path, embedder: Embedder, cache: bool = True) -> KnowledgeBase:
        chunks = load_chunks(kb_dir)
        fingerprint = _fingerprint(chunks, embedder.name)
        cache_path = Path(kb_dir) / INDEX_FILENAME
        if cache and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == fingerprint:
                    vectors = np.asarray(payload["vectors"], dtype=np.float32)
                    logger.info("knowledge base: %d chunks loaded from cache (%s)", len(chunks), embedder.name)
                    return cls(chunks, vectors, embedder)
            except (ValueError, KeyError):
                logger.warning("knowledge base cache unreadable; rebuilding")
        vectors = embedder.embed([c.text for c in chunks])
        if cache:
            cache_path.write_text(
                json.dumps({"fingerprint": fingerprint, "embedder": embedder.name, "vectors": vectors.tolist()}),
                encoding="utf-8",
            )
        logger.info("knowledge base: %d chunks embedded with %s", len(chunks), embedder.name)
        return cls(chunks, vectors, embedder)

    def search(self, query: str, k: int = 6) -> list[tuple[Chunk, float]]:
        query_vector = self.embedder.embed([query])[0]
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


def case_query(case_json: dict[str, Any]) -> str:
    """Turn the pipeline output into a retrieval query (Korean + feature vocabulary of the documents)."""
    seg = case_json.get("segmentation", {}) or {}
    ml = case_json.get("ml_results", {}) or {}
    parts = ["결핵 흉부 X-ray 영상 소견 패턴 위험 평가 감별 진단"]
    label = ml.get("primary_label")
    if label == "no_lesion_detected":
        parts.append("병변 없음 정상 소견 추적 관찰 임상 상관")
    else:
        parts.append("병변 결절 침윤 공동 경화 상엽 질감 불균질 radiomics 텍스처 형태")
    for zone in seg.get("zones", []) or []:
        parts.append(zone)
        if "upper" in zone:
            parts.append("상엽 상부 폐야")
    if (seg.get("num_components") or 0) > 1:
        parts.append("다발성 병변 양측")
    for item in ml.get("top_features", []) or []:
        name = str(item.get("feature", ""))
        if "glcm" in name or "glrlm" in name or "glszm" in name:
            parts.append("질감 heterogeneity texture")
        if "shape" in name:
            parts.append("형태 크기 둘레")
    return " ".join(parts)


def format_context(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{chunk.id}] (score {score:.2f})\n{chunk.text}" for chunk, score in results)
