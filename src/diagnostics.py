from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Compute cosine similarities for one or two batches of vectors."""
    left_2d = _l2_normalize(_as_2d(left))
    right_2d = left_2d if right is None else _l2_normalize(_as_2d(right))
    return left_2d @ right_2d.T


def isolation_rate(left: np.ndarray, right: np.ndarray, threshold: float = 0.1) -> float:
    """Return the fraction of cross-space vector pairs below a cosine threshold."""
    similarities = cosine_similarity(left, right)
    return float(np.mean(similarities < threshold))


def angular_separation_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean angular separation, in degrees, between two vector batches."""
    similarities = np.clip(cosine_similarity(left, right), -1.0, 1.0)
    return float(np.degrees(np.arccos(similarities)).mean())


def space_separation_report(
    vectors_by_org: Mapping[str, np.ndarray],
    *,
    threshold: float = 0.1,
) -> list[Dict[str, Any]]:
    """Compute pairwise cross-organization separation diagnostics."""
    org_ids = list(vectors_by_org)
    report: list[Dict[str, Any]] = []
    for left_index, left_org in enumerate(org_ids):
        for right_org in org_ids[left_index + 1:]:
            left = vectors_by_org[left_org]
            right = vectors_by_org[right_org]
            similarities = cosine_similarity(left, right)
            report.append(
                {
                    "left_org": left_org,
                    "right_org": right_org,
                    "mean_cosine": float(similarities.mean()),
                    "max_cosine": float(similarities.max()),
                    "angular_separation_degrees": angular_separation_degrees(left, right),
                    "isolation_rate": isolation_rate(left, right, threshold=threshold),
                    "threshold": threshold,
                }
            )
    return report


def nearest_neighbor_overlap(original: np.ndarray, transformed: np.ndarray, k: int = 10) -> float:
    """Measure how much top-k neighborhood structure survives transformation."""
    if k <= 0:
        raise ValueError("k must be positive")
    original_vectors = _as_2d(original)
    transformed_vectors = _as_2d(transformed)
    if original_vectors.shape[0] != transformed_vectors.shape[0]:
        raise ValueError("original and transformed must contain the same number of vectors")

    n_vectors = original_vectors.shape[0]
    if n_vectors <= 1:
        return 1.0
    effective_k = min(k, n_vectors - 1)

    original_neighbors = _topk_neighbors(original_vectors, effective_k)
    transformed_neighbors = _topk_neighbors(transformed_vectors, effective_k)
    overlaps = [
        len(set(left).intersection(right)) / effective_k
        for left, right in zip(original_neighbors, transformed_neighbors)
    ]
    return float(np.mean(overlaps))


def _topk_neighbors(vectors: np.ndarray, k: int) -> np.ndarray:
    similarities = cosine_similarity(vectors)
    np.fill_diagonal(similarities, -np.inf)
    return np.argsort(-similarities, axis=1)[:, :k]


def _as_2d(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D vector array; got shape {array.shape}")
    return array


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return vectors / norms
