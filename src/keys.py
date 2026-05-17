from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .vector2trans import KEY_VERSION, Vector2Trans, Vector2TransConfig


REQUIRED_KEY_FIELDS = frozenset(
    {
        "secret_key",
        "transformation_type",
    }
)


def generate_org_key(
    org_id: str,
    dim: int,
    stages: int = 3,
    beta: float = 0.1,
    alpha: float = 0.05,
    secret_key: bytes | str | None = None,
) -> Dict[str, Any]:
    """Generate one organization-specific vector2Trans key."""
    transformer = Vector2Trans(
        Vector2TransConfig(dim=dim, stages=stages, beta=beta, alpha=alpha),
        secret_key=secret_key,
    )
    key = transformer.initialize_key(org_id=org_id)
    key["key_id"] = key_fingerprint(key)
    return key


def validate_key(key: Mapping[str, Any]) -> None:
    """Validate that a mapping contains a usable Trans-RAG organization key."""
    missing = REQUIRED_KEY_FIELDS.difference(key)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"Missing key field(s): {missing_fields}")

    if key["transformation_type"] not in {"vector2trans", "transrag"}:
        raise ValueError(f"Unsupported transformation_type: {key['transformation_type']!r}")

    dim = _first_present(key, "dim", "dim_in", "dim_out")
    if dim is None:
        raise ValueError("Missing key field: dim")
    if int(dim) <= 0:
        raise ValueError("Key field 'dim' must be positive")

    stages = _first_present(key, "stages", default=3)
    beta = _first_present(key, "beta", "nonlinearity_beta", default=0.1)
    alpha = _first_present(key, "alpha", "blinding_scale", default=0.05)
    if int(stages) <= 0:
        raise ValueError("Key field 'stages' must be positive")
    if float(beta) <= 0:
        raise ValueError("Key field 'beta' must be positive")
    if float(alpha) < 0:
        raise ValueError("Key field 'alpha' must be non-negative")

    secret = key["secret_key"]
    secret_bytes = bytes.fromhex(secret) if isinstance(secret, str) else bytes(secret)
    if len(secret_bytes) < 32:
        raise ValueError("Key field 'secret_key' must contain at least 32 bytes")


def key_fingerprint(key: Mapping[str, Any], length: int = 16) -> str:
    """Return a stable non-secret identifier for a key."""
    import hashlib

    secret = key["secret_key"]
    secret_bytes = bytes.fromhex(secret) if isinstance(secret, str) else bytes(secret)
    digest = hashlib.sha256(
        b"transrag-key:"
        + secret_bytes
        + str(key.get("org_id", "")).encode("utf-8")
        + str(_first_present(key, "dim", "dim_in", "dim_out", default="")).encode("ascii")
        + str(_first_present(key, "stages", default="")).encode("ascii")
    ).hexdigest()
    return digest[:length]


def public_key_metadata(key: Mapping[str, Any]) -> Dict[str, Any]:
    """Return shareable key metadata without the secret seed."""
    validate_key(key)
    return {
        "org_id": key.get("org_id"),
        "key_id": key.get("key_id", key_fingerprint(key)),
        "key_version": key.get("key_version", KEY_VERSION),
        "transformation_type": key["transformation_type"],
        "dim": int(_first_present(key, "dim", "dim_in", "dim_out")),
        "stages": int(_first_present(key, "stages", default=3)),
        "beta": float(_first_present(key, "beta", "nonlinearity_beta", default=0.1)),
        "alpha": float(_first_present(key, "alpha", "blinding_scale", default=0.05)),
        "use_permutation": bool(key.get("use_permutation", True)),
        "use_blinding": bool(key.get("use_blinding", True)),
    }


def save_key(key: Mapping[str, Any], path: str | Path) -> None:
    """Save a vector2Trans key as portable JSON."""
    validate_key(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(key), handle, indent=2, sort_keys=True)


def load_key(path: str | Path) -> Dict[str, Any]:
    """Load a vector2Trans key from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        key = json.load(handle)
    validate_key(key)
    return key


def transform_vectors(vectors: np.ndarray, key: Mapping[str, Any]) -> np.ndarray:
    """Transform document or query vectors into one organization's private space."""
    validate_key(key)
    key_dict = dict(key)
    return Vector2Trans.from_key(key_dict).transform(vectors, key_dict)


def transform_query_for_orgs(
    query_vector: np.ndarray,
    org_keys: Mapping[str, Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Transform one query vector for every authorized organization key."""
    return {org_id: transform_vectors(query_vector, key) for org_id, key in org_keys.items()}


def _first_present(values: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return default
