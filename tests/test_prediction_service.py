from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.utils import timezone

from ai_models import services
from ai_models.ml.features import FEATURE_COLUMNS
from ai_models.ml.model import ARTIFACT_DIR, ARTIFACT_NAME, load_bundle
from core.models import Prediction

pytestmark = pytest.mark.django_db


def test_no_observations_gives_no_prediction(airfield, bundle):
    assert services.predict_for_airfield(airfield, bundle=bundle) == []
    assert Prediction.objects.count() == 0


def test_stale_observations_are_refused(airfield, add_obs, bundle):
    add_obs(minutes_ago=300)  # older than MAX_OBSERVATION_AGE_MIN (120)
    assert services.predict_for_airfield(airfield, bundle=bundle) == []


def test_prediction_created_per_horizon(airfield, add_obs, bundle):
    for m in (150, 120, 90, 60, 30, 5):
        add_obs(minutes_ago=m)
    preds = services.predict_for_airfield(airfield, bundle=bundle)
    assert sorted(p.horizon_minutes for p in preds) == [60, 180]
    for p in preds:
        assert 0.0 <= p.thunderstorm_prob <= 1.0
        assert 0.0 <= p.gale_wind_prob <= 1.0
        assert p.model_version == "test-model"
        assert p.confidence is None
        assert p.details["levels"] == {"Thunderstorm": "GREEN", "StrongWind": "GREEN"}
        assert p.details["thresholds"]["Thunderstorm"]["p50"] == 0.5
        assert p.details["features"]["temp_c"] == 30.0


def test_prediction_reacts_to_thunderstorm_and_wind(airfield, add_obs, bundle):
    for m in (150, 120, 90):
        add_obs(minutes_ago=m)
    for m in (60, 30, 5):
        add_obs(minutes_ago=m, thunderstorm=True, wind_speed_ms=10.0, wind_gust_ms=18.0)
    p = services.predict_for_airfield(airfield, bundle=bundle)[0]
    assert p.thunderstorm_prob >= 1 / 3          # at least one of the last 3 hourly buckets has TS
    assert p.gale_wind_prob == pytest.approx(0.9)  # peak wind 18 m/s / 20
    assert p.details["levels"]["StrongWind"] == "RED"


def test_serving_features_match_training_features(airfield, add_obs):
    """The online path (DB rows -> frame) must yield the same feature vector as
    calling the offline feature code directly on the same observations."""
    from ai_models.ml.dataset import build_hourly
    from ai_models.ml.features import build_features

    rng = np.random.default_rng(0)
    for m in range(5, 600, 30):
        add_obs(minutes_ago=m, temperature_c=float(rng.normal(30, 2)),
                pressure_hpa=float(rng.normal(1004, 1)), thunderstorm=bool(rng.random() < 0.3))
    now = timezone.now()
    X, latest = services.latest_feature_row(airfield, now=now)
    assert list(X.columns) == FEATURE_COLUMNS

    obs = services.observations_frame(airfield, now=now)
    expected = build_features(build_hourly(obs), 22.65, 88.45, 5).iloc[[-1]]
    pd.testing.assert_frame_equal(X, expected)
    assert now - latest <= timedelta(minutes=6)


def test_future_observations_are_ignored(airfield, add_obs, bundle):
    add_obs(minutes_ago=10)
    add_obs(minutes_ago=-600, thunderstorm=True, wind_speed_ms=30.0)  # timestamp in the future
    p = services.predict_for_airfield(airfield, bundle=bundle)[0]
    assert p.thunderstorm_prob == 0.0
    assert p.gale_wind_prob == pytest.approx(3.0 / 20)


def test_predict_command_uses_bundle(airfield, add_obs, bundle, monkeypatch, capsys):
    add_obs(minutes_ago=10)
    monkeypatch.setattr("ai_models.management.commands.predict.get_bundle", lambda: bundle)
    call_command("predict")
    assert Prediction.objects.count() == 2
    assert "VECC +60min" in capsys.readouterr().out


@pytest.mark.skipif(not (ARTIFACT_DIR / f"{ARTIFACT_NAME}.joblib").exists(),
                    reason="trained artifact not present")
def test_released_artifact_loads_and_predicts(airfield, add_obs):
    released = load_bundle()
    assert set(released["models"]) == {"thunder_1h", "thunder_3h", "wind_1h", "wind_3h"}
    assert released["feature_columns"] == FEATURE_COLUMNS
    assert released["metadata"]["data"]["sha256"]
    for m in (150, 120, 90, 60, 30, 5):
        add_obs(minutes_ago=m)
    preds = services.predict_for_airfield(airfield, bundle=released)
    assert len(preds) == 2
    for p in preds:
        assert 0.0 <= p.thunderstorm_prob <= 1.0
        assert 0.0 <= p.gale_wind_prob <= 1.0
        assert p.model_version == ARTIFACT_NAME
