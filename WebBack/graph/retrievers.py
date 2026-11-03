"""Retrieval sources for the chat graph beyond the local knowledge base.

- ``SimilarCaseIndex``: nearest reference cases from the Shenzhen evaluation set by standardized radiomics
  features (cosine). These are dataset images, not patients; they contextualise "cases like this one".
- ``PubMedRetriever``: literature evidence through NCBI E-utilities (public, no key needed; an optional
  NCBI_API_KEY raises the rate limit). Fails soft: any error yields an empty list and a warning.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class SimilarCase:
    image_id: str
    label: int
    label_name: str
    similarity: float
    zones: list[str]
    mask_area_fraction: float
    num_components: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimilarCaseIndex:
    def __init__(self, frame: pd.DataFrame, feature_names: list[str]):
        self.frame = frame.reset_index(drop=True)
        self.feature_names = feature_names
        matrix = self.frame[feature_names].to_numpy(dtype=np.float64)
        self.mean = np.nanmean(matrix, axis=0)
        self.std = np.nanstd(matrix, axis=0)
        self.std[self.std == 0] = 1.0
        standardized = np.nan_to_num((matrix - self.mean) / self.std)
        norms = np.linalg.norm(standardized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = standardized / norms

    @classmethod
    def load(cls, csv_path: Path) -> SimilarCaseIndex | None:
        if not Path(csv_path).exists():
            logger.warning("similar-case index: %s not found; retriever disabled", csv_path)
            return None
        frame = pd.read_csv(csv_path)
        if "detected" in frame.columns:
            frame = frame[frame["detected"] == 1]
        features = [c for c in frame.columns if c.startswith("original_")]
        if not features or frame.empty:
            return None
        logger.info("similar-case index: %d reference cases, %d features", len(frame), len(features))
        return cls(frame, features)

    def search(self, features: dict[str, float] | None, k: int = 3) -> list[SimilarCase]:
        if not features:
            return []
        query = np.array([features.get(name, np.nan) for name in self.feature_names], dtype=np.float64)
        query = np.nan_to_num((query - self.mean) / self.std)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scores = self.vectors @ (query / norm)
        order = np.argsort(-scores)[:k]
        results: list[SimilarCase] = []
        for idx in order:
            row = self.frame.iloc[int(idx)]
            label = int(row.get("label", 0) or 0)
            zones = str(row.get("zones", "") or "")
            results.append(
                SimilarCase(
                    image_id=str(row["image_id"]),
                    label=label,
                    label_name="tb" if label == 1 else "normal",
                    similarity=round(float(scores[idx]), 4),
                    zones=[z for z in zones.split("|") if z],
                    mask_area_fraction=round(float(row.get("mask_area_fraction", 0.0) or 0.0), 5),
                    num_components=int(row.get("num_components", 0) or 0),
                )
            )
        return results


@dataclass
class Literature:
    pmid: str
    title: str
    year: str
    journal: str
    abstract: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PubMedRetriever:
    def __init__(self, enabled: bool = True, api_key: str = "", max_results: int = 3, timeout_s: float = 6.0):
        self.enabled = enabled
        self.api_key = api_key
        self.max_results = max_results
        self.timeout_s = timeout_s

    def _params(self, **params: Any) -> dict[str, Any]:
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str) -> list[Literature]:
        if not self.enabled or not query.strip():
            return []
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                search = client.get(
                    f"{EUTILS}/esearch.fcgi",
                    params=self._params(
                        db="pubmed", term=query, retmax=self.max_results, sort="relevance", retmode="json"
                    ),
                )
                search.raise_for_status()
                ids = search.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                fetch = client.get(
                    f"{EUTILS}/efetch.fcgi",
                    params=self._params(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml"),
                )
                fetch.raise_for_status()
            return parse_pubmed_xml(fetch.text)
        except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
            logger.warning("pubmed retrieval failed: %s", exc)
            return []


def parse_pubmed_xml(xml_text: str, abstract_chars: int = 700) -> list[Literature]:
    root = ET.fromstring(xml_text)
    results: list[Literature] = []
    for article in root.iter("PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = " ".join((article.findtext(".//ArticleTitle") or "").split())
        year = (article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "")[:4]
        journal = (article.findtext(".//Journal/Title") or "").strip()
        abstract = " ".join(" ".join(node.itertext()).strip() for node in article.findall(".//AbstractText"))
        if not pmid or not title:
            continue
        results.append(
            Literature(
                pmid=pmid,
                title=title,
                year=year,
                journal=journal,
                abstract=abstract[:abstract_chars],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return results


def literature_query(case_json: dict[str, Any] | None, question: str, classifier_query: str = "") -> str:
    """Compose a PubMed query from the classifier's query, the case and the clinician's question."""
    parts = [classifier_query.strip()] if classifier_query.strip() else ["pulmonary tuberculosis chest radiograph"]
    if case_json:
        ml = case_json.get("ml_results", {}) or {}
        if ml.get("primary_label") == "no_lesion_detected":
            parts.append("normal chest radiograph tuberculosis screening")
        else:
            parts.append("radiomics texture lesion")
        zones = (case_json.get("segmentation", {}) or {}).get("zones", []) or []
        if any("upper" in z for z in zones):
            parts.append("upper lobe")
    return " ".join(parts)[:200]
