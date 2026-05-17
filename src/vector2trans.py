from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from scipy.linalg import qr


KEY_VERSION = "1"


@dataclass(frozen=True)
class Vector2TransConfig:
    dim: int
    stages: int = 3
    beta: float = 0.1
    alpha: float = 0.05
    use_permutation: bool = True
    use_blinding: bool = True


Vector2TransStage = Dict[str, Any]


class TransRAGTransformation:
    """
    Original Trans-RAG vector transformation implementation.

    This is the original local implementation style: parameters and
    permutation patterns are generated during key initialization and serialized
    into the key object.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dim_in = int(config.get("dim_in", 768))
        self.dim_out = int(config.get("dim_out", 768))
        self.name = self.__class__.__name__

        self.stages = int(config.get("stages", 3))
        raw_beta = config.get("nonlinearity_beta", 0.1)
        self.nonlinearity_beta = 0.1 if raw_beta is None else float(raw_beta)

        self.use_permutation = config.get("use_permutation", True)
        self.use_blinding = config.get("use_blinding", True)
        raw_blinding_scale = config.get("blinding_scale", 0.05)
        self.blinding_scale = 0.05 if raw_blinding_scale is None else float(raw_blinding_scale)

        self.secret_key = config.get("secret_key")
        if self.secret_key is None:
            self.secret_key = secrets.token_bytes(32)
        else:
            self.secret_key = _secret_to_bytes(self.secret_key)

        self.parameters = None
        self.permutation_patterns = None
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.dim_in <= 0 or self.dim_out <= 0:
            raise ValueError("dim_in and dim_out must be positive")
        if self.dim_in != self.dim_out:
            raise ValueError("vector2Trans requires dim_in == dim_out")
        if self.stages <= 0:
            raise ValueError("stages must be positive")
        if self.nonlinearity_beta <= 0:
            raise ValueError("nonlinearity_beta must be positive")
        if self.blinding_scale < 0:
            raise ValueError("blinding_scale must be non-negative")
        if len(self.secret_key) < 32:
            raise ValueError("secret_key must contain at least 32 bytes")

    def initialize_key(self) -> Dict[str, Any]:
        self.parameters = self._initialize_parameters()

        if self.use_permutation:
            self.permutation_patterns = self._initialize_permutation_patterns()

        serializable_parameters = []
        for stage_params in self.parameters:
            serializable_stage = {}
            for key, value in stage_params.items():
                if isinstance(value, np.ndarray):
                    serializable_stage[key] = value.tolist()
                else:
                    serializable_stage[key] = value
            serializable_parameters.append(serializable_stage)

        serializable_permutations = None
        if self.use_permutation and self.permutation_patterns:
            serializable_permutations = []
            for pattern in self.permutation_patterns:
                serializable_permutations.append(pattern.tolist())

        key = {
            "parameters": serializable_parameters,
            "secret_key": self.secret_key.hex() if isinstance(self.secret_key, bytes) else self.secret_key,
            "stages": self.stages,
            "nonlinearity_beta": self.nonlinearity_beta,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "transformation_type": "transrag",
            "use_permutation": self.use_permutation,
            "use_blinding": self.use_blinding,
            "blinding_scale": self.blinding_scale,
        }

        if self.use_permutation and serializable_permutations:
            key["permutation_patterns"] = serializable_permutations

        return key

    def _initialize_parameters(self):
        parameters = []

        for stage in range(self.stages):
            stage_seed = hashlib.sha256(self.secret_key + f"stage{stage}".encode()).digest()

            np.random.seed(int.from_bytes(stage_seed[:4], byteorder="big"))

            random_matrix = np.random.randn(self.dim_in, self.dim_out)
            q, r = qr(random_matrix)

            diagonal_signs = np.sign(np.diag(r))
            diagonal_signs[diagonal_signs == 0] = 1
            d = np.diag(diagonal_signs)
            orthogonal_matrix = q @ d

            offset_scale = 0.1 * (stage + 1)
            offset_seed = int.from_bytes(stage_seed[4:8], byteorder="big")
            np.random.seed(offset_seed)
            offset_vector = offset_scale * np.random.randn(self.dim_out)

            norm_seed = int.from_bytes(stage_seed[8:12], byteorder="big")
            np.random.seed(norm_seed)
            norm_vector = 0.05 * np.random.randn(self.dim_out)

            beta_seed = int.from_bytes(stage_seed[12:16], byteorder="big")
            np.random.seed(beta_seed)
            beta = self.nonlinearity_beta * (1 + 0.2 * np.random.randn())

            parameters.append(
                {
                    "orthogonal_matrix": orthogonal_matrix,
                    "offset_vector": offset_vector,
                    "norm_vector": norm_vector,
                    "beta": beta,
                }
            )

        return parameters

    def _initialize_permutation_patterns(self) -> List[np.ndarray]:
        permutation_patterns = []

        for stage in range(self.stages):
            stage_perm_seed = hashlib.sha256(
                self.secret_key + f"perm_stage{stage}".encode()
            ).digest()

            np.random.seed(int.from_bytes(stage_perm_seed[:4], byteorder="big"))

            permutation = np.arange(self.dim_in)
            np.random.shuffle(permutation)

            permutation_patterns.append(permutation)

        return permutation_patterns

    def _generate_blinding_factors(self, vector: np.ndarray, stage_id: int) -> np.ndarray:
        vector_fingerprint = hashlib.sha256(vector.tobytes()).digest()[:8]
        combined_seed = hashlib.sha256(
            self.secret_key
            + f"blind_stage{stage_id}".encode()
            + vector_fingerprint
        ).digest()

        seed = int.from_bytes(combined_seed[:4], byteorder="big")
        random_generator = np.random.RandomState(seed)

        dim = vector.shape[0]
        blinding_factors = random_generator.randn(dim)

        norm = np.linalg.norm(blinding_factors)
        if norm <= 1e-12:
            return np.zeros(dim, dtype=vector.dtype)
        blinding_factors = self.blinding_scale * blinding_factors / norm

        return blinding_factors

    def _apply_permutation(self, vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        return vector[permutation]

    def _apply_inverse_permutation(self, vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        inverse_perm = np.zeros_like(permutation)
        inverse_perm[permutation] = np.arange(len(permutation))
        return vector[inverse_perm.astype(int)]

    def transform(self, vectors: np.ndarray, key: Dict[str, Any]) -> np.ndarray:
        if self.parameters is None or key.get("secret_key") != getattr(self, "secret_key", None):
            self.secret_key = _secret_to_bytes(key["secret_key"])
            self.stages = int(key.get("stages", 3))
            raw_beta = key.get("nonlinearity_beta", 0.1)
            self.nonlinearity_beta = 0.1 if raw_beta is None else float(raw_beta)
            self.dim_in = int(_first_present(key, "dim_in", "dim", default=self.dim_in))
            self.dim_out = int(_first_present(key, "dim_out", "dim", default=self.dim_out))

            self.use_permutation = key.get("use_permutation", True)
            self.use_blinding = key.get("use_blinding", True)
            raw_blinding_scale = key.get("blinding_scale", 0.05)
            self.blinding_scale = 0.05 if raw_blinding_scale is None else float(raw_blinding_scale)
            self._validate_configuration()

            if "parameters" in key:
                parameters = key["parameters"]
                self.parameters = []
                for stage_params in parameters:
                    restored_stage = {}
                    for param_name, value in stage_params.items():
                        if isinstance(value, list):
                            restored_stage[param_name] = np.array(value)
                        else:
                            restored_stage[param_name] = value
                    self.parameters.append(restored_stage)
            else:
                self.parameters = self._initialize_parameters()

            if self.use_permutation and "permutation_patterns" in key:
                self.permutation_patterns = []
                for pattern in key["permutation_patterns"]:
                    self.permutation_patterns.append(np.array(pattern))
            elif self.use_permutation:
                self.permutation_patterns = self._initialize_permutation_patterns()

        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        else:
            vectors = np.asarray(vectors)
            if not np.issubdtype(vectors.dtype, np.floating):
                vectors = vectors.astype(np.float32)

        single_vector = False
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            single_vector = True
        elif vectors.ndim != 2:
            raise ValueError(f"Expected a 1-D or 2-D vector array; got shape {vectors.shape}")

        if vectors.shape[1] != self.dim_in:
            raise ValueError(f"Expected vectors with dimension {self.dim_in}; got {vectors.shape[1]}")

        transformed_vectors = np.zeros((vectors.shape[0], self.dim_out), dtype=vectors.dtype)

        for index in range(vectors.shape[0]):
            vector = vectors[index]
            transformed = vector
            for stage in range(len(self.parameters)):
                if self.use_permutation and self.permutation_patterns:
                    transformed = self._apply_permutation(transformed, self.permutation_patterns[stage])

                if self.use_blinding:
                    pre_blinding = self._generate_blinding_factors(transformed, stage)
                    transformed = transformed + pre_blinding - np.mean(pre_blinding)

                transformed = self._apply_stage_transformation(transformed, self.parameters[stage])

                if self.use_blinding:
                    post_blinding = self._generate_blinding_factors(transformed, stage + self.stages)
                    transformed = transformed + post_blinding - np.mean(post_blinding)

                if self.use_permutation and self.permutation_patterns:
                    transformed = self._apply_inverse_permutation(transformed, self.permutation_patterns[stage])

            transformed_vectors[index] = transformed

        normalized = _l2_normalize(transformed_vectors)

        if single_vector:
            return normalized[0]
        return normalized

    def _nonlinear_function(self, x, beta):
        if abs(beta) <= 1e-12:
            return x
        return np.tanh(beta * x) / beta

    def _apply_stage_transformation(self, vector, stage_params):
        w = stage_params["orthogonal_matrix"]
        b = stage_params["offset_vector"]
        c = stage_params["norm_vector"]
        beta = stage_params["beta"]

        vector_with_offset = vector + b
        nonlinear_output = self._nonlinear_function(vector_with_offset, beta)
        transformed = nonlinear_output @ w
        transformed = transformed + c

        return transformed


class Vector2Trans(TransRAGTransformation):
    """Compatibility wrapper around the original TransRAGTransformation."""

    def __init__(self, config: Vector2TransConfig | Mapping[str, Any], secret_key: bytes | str | None = None):
        legacy_config = _config_to_legacy_dict(config)
        if secret_key is not None:
            legacy_config["secret_key"] = _secret_to_bytes(secret_key)
        super().__init__(legacy_config)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Vector2Trans":
        return cls(config=config, secret_key=config.get("secret_key"))

    @classmethod
    def from_key(cls, key: Mapping[str, Any]) -> "Vector2Trans":
        return cls(config=key, secret_key=key["secret_key"])

    def initialize_key(self, org_id: str | None = None) -> Dict[str, Any]:
        key = super().initialize_key()
        key["key_version"] = KEY_VERSION
        key["dim"] = key["dim_in"]
        if org_id is not None:
            key["org_id"] = org_id
        return key

    def transform(self, vectors: np.ndarray, key: Dict[str, Any] | None = None) -> np.ndarray:
        if key is None:
            key = self.initialize_key()
        return super().transform(vectors, dict(key))


def _config_to_legacy_dict(config: Vector2TransConfig | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(config, Vector2TransConfig):
        return {
            "dim_in": config.dim,
            "dim_out": config.dim,
            "stages": config.stages,
            "nonlinearity_beta": config.beta,
            "use_permutation": config.use_permutation,
            "use_blinding": config.use_blinding,
            "blinding_scale": config.alpha,
        }

    dim = _first_present(config, "dim", "dim_in", "dim_out", default=768)
    return {
        "dim_in": int(_first_present(config, "dim_in", "dim", default=dim)),
        "dim_out": int(_first_present(config, "dim_out", "dim", default=dim)),
        "stages": int(_first_present(config, "stages", default=3)),
        "nonlinearity_beta": float(_first_present(config, "nonlinearity_beta", "beta", default=0.1)),
        "use_permutation": bool(_first_present(config, "use_permutation", default=True)),
        "use_blinding": bool(_first_present(config, "use_blinding", default=True)),
        "blinding_scale": float(_first_present(config, "blinding_scale", "alpha", default=0.05)),
        "secret_key": _secret_to_bytes(config["secret_key"]) if "secret_key" in config else None,
    }


def _first_present(values: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return default


def _secret_to_bytes(secret_key: bytes | str) -> bytes:
    if isinstance(secret_key, bytes):
        return secret_key
    try:
        return bytes.fromhex(secret_key)
    except ValueError:
        return secret_key.encode("utf-8")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1e-10)
    return vectors / norms


vector2Trans = Vector2Trans
