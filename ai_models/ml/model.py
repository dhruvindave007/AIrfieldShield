"""Model wrapper and artifact I/O shared by training and serving."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_NAME = "nowcast_hgb_v1"

# Alert levels are tied to precision targets chosen on the validation year.
LEVEL_BY_PRECISION = (("RED", "p75"), ("ORANGE", "p50"), ("YELLOW", "p25"))


class CalibratedNowcaster:
    """A fitted classifier plus an isotonic calibrator fitted on held-out
    validation data. ``predict_proba`` returns calibrated event probabilities."""

    def __init__(self, classifier, calibrator, feature_columns, thresholds=None):
        self.classifier = classifier
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns)
        self.thresholds = dict(thresholds or {})

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X[self.feature_columns])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_proba(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def level(self, prob: float) -> str:
        return level_for(prob, self.thresholds)


def level_for(prob, thresholds: dict) -> str:
    """Map a calibrated probability to GREEN/YELLOW/ORANGE/RED using the
    per-target thresholds ({"p25": t, "p50": t, "p75": t}; None = unattainable)."""
    if prob is None:
        return "GREEN"
    for level, key in LEVEL_BY_PRECISION:
        thr = (thresholds or {}).get(key)
        if thr is not None and prob >= thr:
            return level
    return "GREEN"


def save_bundle(bundle: dict, metadata: dict, directory: Path = ARTIFACT_DIR,
                name: str = ARTIFACT_NAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(bundle, path, compress=3)
    (directory / f"{name}.json").write_text(json.dumps(_jsonable(metadata), indent=2),
                                            encoding="utf-8")
    return path


def _jsonable(obj):
    """Plain JSON types only; NaN becomes null so the file is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, str)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return str(obj)


def load_bundle(directory: Path = ARTIFACT_DIR, name: str = ARTIFACT_NAME) -> dict:
    """Load {'models': {target: CalibratedNowcaster}, 'feature_columns', 'metadata'}."""
    path = Path(directory) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact {path} not found. Train it with: python -m ai_models.ml.train")
    bundle = joblib.load(path)
    meta_path = Path(directory) / f"{name}.json"
    bundle["metadata"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return bundle
