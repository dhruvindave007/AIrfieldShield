from datetime import timedelta

import numpy as np
import pytest
from django.utils import timezone

from ai_models.ml.features import FEATURE_COLUMNS
from ai_models.ml.model import CalibratedNowcaster


class RuleClassifier:
    """Deterministic stand-in for a fitted classifier, used only in unit tests:
    probability is driven by one feature so tests can steer the output."""

    def __init__(self, feature, scale):
        self.feature, self.scale = feature, scale

    def predict_proba(self, X):
        p = np.clip(np.nan_to_num(X[self.feature].to_numpy(dtype=float)) / self.scale, 0.0, 1.0)
        return np.column_stack([1 - p, p])


THRESHOLDS = {"p25": 0.2, "p50": 0.5, "p75": 0.8}


@pytest.fixture
def bundle():
    """Thunder probability = ts_sum3h / 3, wind probability = peak_wind_ms / 20."""
    def model(feature, scale):
        return CalibratedNowcaster(RuleClassifier(feature, scale), None, FEATURE_COLUMNS, THRESHOLDS)
    return {
        "models": {"thunder_1h": model("ts_sum3h", 3.0), "thunder_3h": model("ts_sum3h", 3.0),
                   "wind_1h": model("peak_wind_ms", 20.0), "wind_3h": model("peak_wind_ms", 20.0)},
        "feature_columns": FEATURE_COLUMNS,
        "metadata": {"artifact": "test-model"},
    }


@pytest.fixture
def airfield(db):
    from core.models import Airfield, WeatherStation
    af = Airfield.objects.create(name="Kolkata", icao="VECC", latitude=22.65, longitude=88.45,
                                 elevation_m=5)
    WeatherStation.objects.create(airfield=af, station_id="VECC", latitude=22.65, longitude=88.45)
    return af


@pytest.fixture
def add_obs(airfield):
    """Factory: add_obs(minutes_ago, **fields) creates one observation."""
    from core.models import WeatherObservation
    station = airfield.stations.first()
    now = timezone.now()

    def _add(minutes_ago, **fields):
        defaults = dict(temperature_c=30.0, dewpoint_c=24.0, humidity=70.0, pressure_hpa=1004.0,
                        wind_speed_ms=3.0, wind_dir_deg=180.0, visibility_km=5.0,
                        thunderstorm=False, cumulonimbus=False, precipitation=False,
                        trend_thunderstorm=False)
        defaults.update(fields)
        return WeatherObservation.objects.create(
            station=station, timestamp=now - timedelta(minutes=minutes_ago), **defaults)
    return _add
