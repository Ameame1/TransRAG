from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

from .keyring import KeyRing
from .keys import transform_vectors


@dataclass(frozen=True)
class QueryPlan:
    """Organization-specific transformed queries for one plaintext query vector."""

    authorized_org_ids: tuple[str, ...]
    transformed_queries: Dict[str, np.ndarray]

    def query_for(self, org_id: str) -> np.ndarray:
        try:
            return self.transformed_queries[org_id]
        except KeyError as exc:
            raise KeyError(f"Query was not transformed for organization: {org_id}") from exc


@dataclass(frozen=True)
class RetrievalResult:
    """A vector-store result after Trans-RAG score aggregation."""

    org_id: str
    item_id: str
    score: float
    raw_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))
        raw_score = self.score if self.raw_score is None else float(self.raw_score)
        object.__setattr__(self, "raw_score", raw_score)


SearchHit = RetrievalResult | Mapping[str, Any] | tuple[Any, float]
SearchCallback = Callable[[str, np.ndarray, int], Sequence[SearchHit]]


class TransRAGCore:
    """
    Core Trans-RAG workflow without binding to a specific vector database.

    The caller owns embedding, indexing, retrieval, and LLM generation. This
    class owns organization key selection, vector2Trans application, authorized
    query planning, and cross-organization result aggregation.
    """

    def __init__(
        self,
        org_keys: KeyRing | Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.keyring = org_keys if isinstance(org_keys, KeyRing) else KeyRing(org_keys)

    def transform_documents(self, org_id: str, document_vectors: np.ndarray) -> np.ndarray:
        """Transform one organization's document vectors before external indexing."""
        return transform_vectors(document_vectors, self.keyring.get(org_id))

    def transform_documents_by_org(self, vectors_by_org: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform document vectors for multiple organizations."""
        return {org_id: self.transform_documents(org_id, vectors) for org_id, vectors in vectors_by_org.items()}

    def build_query_plan(
        self,
        query_vector: np.ndarray,
        authorized_org_ids: Iterable[str] | None = None,
    ) -> QueryPlan:
        """Transform one query into every authorized organization's private space."""
        org_keys = self.keyring.select(authorized_org_ids)
        transformed_queries = {
            org_id: transform_vectors(query_vector, key)
            for org_id, key in org_keys.items()
        }
        return QueryPlan(
            authorized_org_ids=tuple(org_keys),
            transformed_queries=transformed_queries,
        )

    def search(
        self,
        query_vector: np.ndarray,
        search_fn: SearchCallback,
        *,
        authorized_org_ids: Iterable[str] | None = None,
        per_org_top_k: int = 10,
        final_top_k: int | None = 10,
        score_normalization: str = "minmax",
    ) -> list[RetrievalResult]:
        """
        Transform an authorized query, call user-owned vector stores, and rank results.

        `search_fn` receives `(org_id, transformed_query, top_k)` and returns
        hits as `RetrievalResult`, mappings with `item_id`/`score`, or
        `(item_id, score)` tuples.
        """
        if per_org_top_k <= 0:
            raise ValueError("per_org_top_k must be positive")
        if final_top_k is not None and final_top_k <= 0:
            raise ValueError("final_top_k must be positive when provided")

        plan = self.build_query_plan(query_vector, authorized_org_ids)
        results_by_org: Dict[str, list[RetrievalResult]] = {}
        for org_id in plan.authorized_org_ids:
            raw_hits = search_fn(org_id, plan.query_for(org_id), per_org_top_k)
            results_by_org[org_id] = [
                _coerce_result(hit, org_id=org_id)
                for hit in raw_hits
            ]

        return aggregate_results(
            results_by_org,
            top_k=final_top_k,
            score_normalization=score_normalization,
        )


def aggregate_results(
    results_by_org: Mapping[str, Sequence[RetrievalResult]],
    *,
    top_k: int | None = 10,
    score_normalization: str = "minmax",
) -> list[RetrievalResult]:
    """Normalize per-organization scores and return a global ranking."""
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if score_normalization not in {"none", "minmax", "zscore", "rank"}:
        raise ValueError("score_normalization must be one of: none, minmax, zscore, rank")

    normalized: list[RetrievalResult] = []
    for org_id, results in results_by_org.items():
        org_results = [_coerce_result(result, org_id=org_id) for result in results]
        normalized.extend(_normalize_org_results(org_results, method=score_normalization))

    normalized.sort(key=lambda result: result.score, reverse=True)
    return normalized if top_k is None else normalized[:top_k]


def _normalize_org_results(results: Sequence[RetrievalResult], *, method: str) -> list[RetrievalResult]:
    if method == "none" or not results:
        return list(results)

    scores = np.asarray([result.raw_score for result in results], dtype=np.float64)
    if method == "minmax":
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score <= min_score:
            scaled = np.ones_like(scores, dtype=np.float64)
        else:
            scaled = (scores - min_score) / (max_score - min_score)
    elif method == "zscore":
        std = float(scores.std())
        scaled = np.zeros_like(scores, dtype=np.float64) if std <= 1e-12 else (scores - float(scores.mean())) / std
    else:
        order = np.argsort(-scores)
        scaled = np.zeros_like(scores, dtype=np.float64)
        count = len(scores)
        for rank, index in enumerate(order):
            scaled[index] = (count - rank) / count

    return [replace(result, score=float(score)) for result, score in zip(results, scaled)]


def _coerce_result(
    hit: RetrievalResult | Mapping[str, Any] | tuple[Any, float],
    *,
    org_id: str,
) -> RetrievalResult:
    if isinstance(hit, RetrievalResult):
        return hit if hit.org_id == org_id else replace(hit, org_id=org_id)

    if isinstance(hit, tuple):
        item_id, score = hit
        return RetrievalResult(org_id=org_id, item_id=str(item_id), score=float(score), raw_score=float(score))

    if isinstance(hit, MappingABC):
        if "score" not in hit:
            raise ValueError("Result mapping must include a 'score' field")
        item_id = hit.get("item_id", hit.get("id"))
        if item_id is None:
            raise ValueError("Result mapping must include 'item_id' or 'id'")
        score = float(hit["score"])
        raw_score = float(hit.get("raw_score", score))
        metadata = {
            key: value
            for key, value in hit.items()
            if key not in {"org_id", "item_id", "id", "score", "raw_score"}
        }
        return RetrievalResult(
            org_id=str(hit.get("org_id", org_id)),
            item_id=str(item_id),
            score=score,
            raw_score=raw_score,
            metadata=metadata,
        )

    raise TypeError(f"Unsupported result type: {type(hit)!r}")
