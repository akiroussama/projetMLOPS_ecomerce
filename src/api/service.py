import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .schemas import PredictRequest

# ---------------------------------------------------------------------------
# Pickle compatibility: the TF-IDF vectorizer was pickled with
# ``preprocessor=normalize_text`` from build_features.  If that module was
# executed as ``__main__`` when the pickle was created, the reference stored
# inside the file points to ``__main__.normalize_text``.  We patch __main__
# so ``pickle.load`` can resolve the symbol regardless of how it was saved.
# ---------------------------------------------------------------------------
try:
    from ..features.build_features import normalize_text as _normalize_text
except Exception:
    try:
        from src.features.build_features import normalize_text as _normalize_text
    except Exception:
        _normalize_text = None

if _normalize_text is not None:
    import __main__ as _main_module
    if not hasattr(_main_module, "normalize_text"):
        _main_module.normalize_text = _normalize_text


class ModelNotReadyError(RuntimeError):
    pass


class PredictionExecutionError(RuntimeError):
    pass


@dataclass
class ArtifactPaths:
    model_path: Optional[Path]
    vectorizer_path: Optional[Path]
    label_mapping_path: Optional[Path]


@dataclass
class PredictionService:
    project_root: Path = Path(__file__).resolve().parents[2]

    def __post_init__(self) -> None:
        self.model: Any = None
        self.vectorizer: Any = None
        self.label_mapping: Optional[Dict[str, Any]] = None
        self.load_error: Optional[str] = None
        self.artifacts = self._resolve_artifact_paths()

    def _resolve_artifact_paths(self) -> ArtifactPaths:
        model_dir = Path(os.getenv("MODEL_DIR", self.project_root / "models"))

        return ArtifactPaths(
            model_path=self._resolve_artifact_path(
                model_dir,
                env_var="MODEL_FILE",
                candidates=("baseline_model.pkl", "classifier.pkl", "model.pkl"),
            ),
            vectorizer_path=self._resolve_artifact_path(
                model_dir,
                env_var="VECTORIZER_FILE",
                candidates=("tfidf_vectorizer.pkl",),
            ),
            label_mapping_path=self._resolve_artifact_path(
                model_dir,
                env_var="LABEL_MAPPING_FILE",
                candidates=(
                    "label_mapping.json",
                    "mapper.json",
                    "label_mapping.pkl",
                    "modalite_mapping.pkl",
                    "mapping.pkl",
                ),
            ),
        )

    @staticmethod
    def _resolve_artifact_path(
        model_dir: Path, env_var: str, candidates: tuple[str, ...]
    ) -> Optional[Path]:
        override = os.getenv(env_var)
        if override:
            return Path(override)

        for candidate in candidates:
            candidate_path = model_dir / candidate
            if candidate_path.exists():
                return candidate_path

        return None

    def load(self) -> bool:
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        self.load_error = None
        self.artifacts = self._resolve_artifact_paths()

        if self.artifacts.model_path is None:
            self.load_error = (
                "No serialized model artifact was found. "
                "Expected one of baseline_model.pkl, classifier.pkl or model.pkl."
            )
            return False

        try:
            with self.artifacts.model_path.open("rb") as model_file:
                self.model = pickle.load(model_file)

            if self.artifacts.vectorizer_path is not None:
                with self.artifacts.vectorizer_path.open("rb") as vectorizer_file:
                    self.vectorizer = pickle.load(vectorizer_file)

            if self.artifacts.label_mapping_path is not None:
                self.label_mapping = self._load_label_mapping(
                    self.artifacts.label_mapping_path
                )
        except Exception as exc:  # pragma: no cover - exercised through health detail
            self.load_error = f"Failed to load model artifacts: {exc}"
            self.model = None
            self.vectorizer = None
            self.label_mapping = None
            return False

        return True

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.is_ready else "degraded",
            "model_loaded": self.is_ready,
            "model_name": type(self.model).__name__ if self.model is not None else None,
            "model_path": self._stringify_path(self.artifacts.model_path),
            "vectorizer_path": self._stringify_path(self.artifacts.vectorizer_path),
            "label_mapping_path": self._stringify_path(
                self.artifacts.label_mapping_path
            ),
            "detail": self.load_error or "Model artifacts loaded successfully.",
        }

    def predict(self, payload: PredictRequest) -> Dict[str, Any]:
        if not self.is_ready:
            raise ModelNotReadyError(
                self.load_error or "Prediction service is not ready yet."
            )

        try:
            model_input = self._build_model_input(payload)
            raw_prediction = self.model.predict(model_input)[0]
            predicted_code = self._to_native(raw_prediction)
            predicted_label = self._map_label(predicted_code)
            confidence = self._predict_confidence(model_input)
        except Exception as exc:
            raise PredictionExecutionError("Prediction pipeline failed.") from exc

        return {
            "predicted_label": str(predicted_label),
            "predicted_code": predicted_code,
            "confidence": confidence,
            "model_name": type(self.model).__name__,
            "productid": payload.productid,
            "imageid": payload.imageid,
        }

    def _build_model_input(self, payload: PredictRequest) -> Any:
        text = self._compose_text(payload.designation, payload.description)

        if self.vectorizer is not None:
            return self.vectorizer.transform([text])

        return [text]

    @staticmethod
    def _compose_text(designation: str, description: Optional[str]) -> str:
        return " ".join(part.strip() for part in [designation, description or ""] if part)

    def _predict_confidence(self, model_input: Any) -> Optional[float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(model_input)[0]
            return round(float(np.max(probabilities)), 6)

        return None

    def _map_label(self, predicted_code: Any) -> Any:
        if not self.label_mapping:
            return predicted_code

        return self.label_mapping.get(str(predicted_code), predicted_code)

    @staticmethod
    def _stringify_path(path: Optional[Path]) -> Optional[str]:
        return str(path) if path is not None else None

    @staticmethod
    def _to_native(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _load_label_mapping(path: Path) -> Dict[str, Any]:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as mapping_file:
                return json.load(mapping_file)

        with path.open("rb") as mapping_file:
            raw_mapping = pickle.load(mapping_file)

        return {str(key): value for key, value in raw_mapping.items()}
