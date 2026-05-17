from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from .keys import generate_org_key, load_key, public_key_metadata, save_key, validate_key


class KeyRing:
    """In-memory registry for organization-specific Trans-RAG keys."""

    def __init__(
        self,
        keys: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
        *,
        require_same_dim: bool = True,
    ) -> None:
        self.require_same_dim = require_same_dim
        self._keys: Dict[str, Dict[str, Any]] = {}

        if keys is None:
            return
        if isinstance(keys, Mapping):
            for org_id, key in keys.items():
                self.add_key(key, org_id=str(org_id))
        else:
            for key in keys:
                self.add_key(key)

    @classmethod
    def generate(
        cls,
        org_ids: Iterable[str],
        *,
        dim: int,
        stages: int = 3,
        beta: float = 0.1,
        alpha: float = 0.05,
    ) -> "KeyRing":
        ring = cls()
        for org_id in org_ids:
            ring.add_key(generate_org_key(org_id, dim=dim, stages=stages, beta=beta, alpha=alpha))
        return ring

    @classmethod
    def load_dir(cls, directory: str | Path, pattern: str = "*.json") -> "KeyRing":
        ring = cls()
        for path in sorted(Path(directory).glob(pattern)):
            ring.add_key(load_key(path))
        return ring

    def add_key(self, key: Mapping[str, Any], *, org_id: str | None = None, replace: bool = False) -> None:
        validate_key(key)
        resolved_org_id = org_id or key.get("org_id")
        if resolved_org_id is None:
            raise ValueError("Organization key must include org_id or be added with org_id=")
        resolved_org_id = str(resolved_org_id)

        if resolved_org_id in self._keys and not replace:
            raise ValueError(f"Organization key already exists: {resolved_org_id}")

        if self.require_same_dim:
            self._validate_dimension(key)

        stored = dict(key)
        stored["org_id"] = resolved_org_id
        self._keys[resolved_org_id] = stored

    def get(self, org_id: str) -> Dict[str, Any]:
        try:
            return dict(self._keys[org_id])
        except KeyError as exc:
            raise KeyError(f"Unknown organization: {org_id}") from exc

    def select(self, org_ids: Iterable[str] | None = None) -> Dict[str, Dict[str, Any]]:
        selected_ids = list(self._keys) if org_ids is None else list(org_ids)
        return {org_id: self.get(org_id) for org_id in selected_ids}

    def metadata(self) -> Dict[str, Dict[str, Any]]:
        return {org_id: public_key_metadata(key) for org_id, key in self._keys.items()}

    def save_dir(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for org_id, key in self._keys.items():
            save_key(key, directory / f"{_safe_filename(org_id)}.key.json")

    @property
    def dim(self) -> int | None:
        if not self._keys:
            return None
        first_key = next(iter(self._keys.values()))
        return public_key_metadata(first_key)["dim"]

    @property
    def org_ids(self) -> tuple[str, ...]:
        return tuple(self._keys)

    def __contains__(self, org_id: object) -> bool:
        return org_id in self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def _validate_dimension(self, key: Mapping[str, Any]) -> None:
        if not self._keys:
            return
        expected_dim = self.dim
        actual_dim = public_key_metadata(key)["dim"]
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(f"All keys in this KeyRing must use dim={expected_dim}; got {actual_dim}")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
