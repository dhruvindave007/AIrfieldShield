"""Online prediction service: observations in the DB -> features -> calibrated
probabilities -> Prediction rows. Uses exactly the feature code used in training."""
from __future__ import annotations

import logging
import math
from datetime import timedelta
from functools import lru_cache

import pandas as pd
from django.conf import settings
from django.utils import timezone

from core.models import Airfield, Prediction, WeatherObservation

from .ml.dataset import build_hourly
from .ml.features import build_features
from .ml.model import load_bundle

logger = logging.getLogger(__name__)

HISTORY_HOURS = 30  # longest feature window is 24 h
SNAPSHOT_FEATURES = ("temp_c", "dewpoint_c", "pressure_hpa", "peak_wind_ms", "ts", "cb",
                     "trend_ts", "pressure_d3h", "ts_sum3h", "n_reports")

_RENAME = {
    "timestamp": "valid", "temperature_c": "temp_c", "humidity": "rh",
    "wind_speed_ms": "wind_ms", "wind_gust_ms": "gust_ms", "wind_dir_deg": "wind_dir",
    "thunderstorm": "ts", "cumulonimbus": "cb", "precipitation": "precip",
    "trend_thunderstorm": "trend_ts",
}


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()


def observations_frame(airfield: Airfield, now=None, hours: int = HISTORY_HOURS) -> pd.DataFrame:
    """Recent observations of an airfield as canonical columns (see ml.dataset)."""
    now = now or timezone.now()
    qs = WeatherObservation.objects.filter(
        station__airfield=airfield, timestamp__gt=now - timedelta(hours=hours), timestamp__lte=now,
    ).order_by("timestamp").values(
        "timestamp", "temperature_c", "dewpoint_c", "humidity", "pressure_hpa", "wind_speed_ms",
        "wind_gust_ms", "wind_dir_deg", "visibility_km", "thunderstorm", "cumulonimbus",
        "precipitation", "trend_thunderstorm")
    df = pd.DataFrame.from_records(list(qs))
    if df.empty:
        return df
    df = df.rename(columns=_RENAME)
    cols = [c for c in df.columns if c != "valid"]
    df[cols] = df[cols].astype(float)  # None/bool -> NaN/0.0/1.0
    return df


def latest_feature_row(airfield: Airfield, now=None):
    """(one-row feature frame, latest observation time) for the newest hourly
    bucket, or None if the airfield has no sufficiently recent observation."""
    now = now or timezone.now()
    obs = observations_frame(airfield, now=now)
    if obs.empty:
        return None
    latest = pd.Timestamp(obs["valid"].max())
    max_age = timedelta(minutes=settings.AIRFIELDSHIELD["MAX_OBSERVATION_AGE_MIN"])
    if now - latest > max_age:
        return None
    hourly = build_hourly(obs)
    feats = build_features(hourly, airfield.latitude or 0.0, airfield.longitude or 0.0,
                           airfield.elevation_m or 0.0)
    return feats.iloc[[-1]], latest


def predict_for_airfield(airfield: Airfield, bundle: dict | None = None, now=None) -> list[Prediction]:
    """Create one Prediction per horizon. Returns [] when observations are missing or stale."""
    bundle = bundle or get_bundle()
    now = now or timezone.now()
    row = latest_feature_row(airfield, now=now)
    if row is None:
        logger.warning("No recent observations for %s; skipping prediction", airfield)
        return []
    X, latest = row
    models = bundle["models"]
    version = bundle.get("metadata", {}).get("artifact", "unknown")
    values = X.iloc[0]
    snapshot = {k: (None if math.isnan(values[k]) else round(float(values[k]), 2))
                for k in SNAPSHOT_FEATURES}
    completeness = float(values.notna().mean())

    out = []
    for horizon in settings.AIRFIELDSHIELD["HORIZONS_MIN"]:
        h = horizon // 60
        thunder, wind = models.get(f"thunder_{h}h"), models.get(f"wind_{h}h")
        if thunder is None and wind is None:
            continue
        p_th = float(thunder.predict_proba(X)[0]) if thunder else None
        p_w = float(wind.predict_proba(X)[0]) if wind else None
        details = {
            "latest_observation": latest.isoformat(),
            "feature_completeness": round(completeness, 3),
            "features": snapshot,
            "thresholds": {
                "Thunderstorm": thunder.thresholds if thunder else None,
                "StrongWind": wind.thresholds if wind else None,
            },
            "levels": {
                "Thunderstorm": thunder.level(p_th) if thunder else "GREEN",
                "StrongWind": wind.level(p_w) if wind else "GREEN",
            },
        }
        out.append(Prediction.objects.create(
            airfield=airfield, horizon_minutes=horizon,
            thunderstorm_prob=None if p_th is None else round(p_th, 4),
            gale_wind_prob=None if p_w is None else round(p_w, 4),
            confidence=None,  # no separate confidence estimate; probabilities are calibrated
            details=details, model_version=version))
    return out


def predict_all(now=None) -> list[Prediction]:
    bundle = get_bundle()
    preds = []
    for airfield in Airfield.objects.all():
        preds.extend(predict_for_airfield(airfield, bundle=bundle, now=now))
    return preds
