from datetime import timedelta

import pytest
from django.utils import timezone

from ai_models.ml.model import level_for
from alerts.models import Alert
from alerts.services import evaluate_prediction, generate_alerts, level_for_prediction
from core.models import Prediction

pytestmark = pytest.mark.django_db

T = {"p25": 0.2, "p50": 0.5, "p75": 0.8}


def make_pred(airfield, thunder=0.0, wind=0.0, horizon=60, thresholds=T, age_min=0):
    details = {"thresholds": {"Thunderstorm": thresholds, "StrongWind": thresholds}} if thresholds else {}
    p = Prediction.objects.create(airfield=airfield, horizon_minutes=horizon,
                                  thunderstorm_prob=thunder, gale_wind_prob=wind,
                                  details=details, model_version="test-model")
    if age_min:
        Prediction.objects.filter(pk=p.pk).update(created_at=timezone.now() - timedelta(minutes=age_min))
        p.refresh_from_db()
    return p


@pytest.mark.parametrize("prob,expected", [
    (None, "GREEN"), (0.0, "GREEN"), (0.1999, "GREEN"),
    (0.2, "YELLOW"), (0.4999, "YELLOW"),
    (0.5, "ORANGE"), (0.7999, "ORANGE"),
    (0.8, "RED"), (1.0, "RED"),
])
def test_level_boundaries(prob, expected):
    assert level_for(prob, T) == expected


def test_unattainable_threshold_disables_that_level():
    # the model never reached 75% precision on validation -> RED can never fire
    t = {"p25": 0.2, "p50": 0.5, "p75": None}
    assert level_for(0.99, t) == "ORANGE"
    assert level_for(0.99, {"p25": None, "p50": None, "p75": None}) == "GREEN"
    assert level_for(0.99, {}) == "GREEN"


def test_fallback_thresholds_when_prediction_has_none(airfield):
    p = make_pred(airfield, thunder=0.6, thresholds=None)
    assert level_for_prediction(p, "Thunderstorm") == "ORANGE"   # settings fallback 0.25/0.5/0.75


def test_green_prediction_creates_no_alert(airfield):
    assert evaluate_prediction(make_pred(airfield, thunder=0.1, wind=0.05)) == 0
    assert Alert.objects.count() == 0


def test_alert_created_with_level_and_validity(airfield):
    p = make_pred(airfield, thunder=0.6, wind=0.1)
    assert evaluate_prediction(p) == 1
    a = Alert.objects.get()
    assert (a.alert_type, a.level) == ("Thunderstorm", "ORANGE")
    assert a.valid_until == p.created_at + timedelta(minutes=60)
    assert a.meta["prediction_id"] == p.id
    assert "60%" in a.message


def test_both_hazards_alert_independently(airfield):
    assert evaluate_prediction(make_pred(airfield, thunder=0.9, wind=0.3)) == 2
    levels = dict(Alert.objects.values_list("alert_type", "level"))
    assert levels == {"Thunderstorm": "RED", "StrongWind": "YELLOW"}


def test_repeated_predictions_do_not_duplicate(airfield):
    for _ in range(5):
        evaluate_prediction(make_pred(airfield, thunder=0.6))
    assert Alert.objects.count() == 1


def test_escalation_updates_unacknowledged_alert_in_place(airfield):
    evaluate_prediction(make_pred(airfield, thunder=0.3))
    assert evaluate_prediction(make_pred(airfield, thunder=0.9)) == 1
    assert Alert.objects.count() == 1
    assert Alert.objects.get().level == "RED"


def test_lower_level_does_not_downgrade(airfield):
    evaluate_prediction(make_pred(airfield, thunder=0.9))
    assert evaluate_prediction(make_pred(airfield, thunder=0.3)) == 0
    assert Alert.objects.get().level == "RED"


def test_acknowledged_alert_is_not_recreated_but_escalation_is(airfield):
    evaluate_prediction(make_pred(airfield, thunder=0.6))
    Alert.objects.update(acknowledged=True)
    assert evaluate_prediction(make_pred(airfield, thunder=0.6)) == 0   # same level: stay quiet
    assert Alert.objects.count() == 1
    assert evaluate_prediction(make_pred(airfield, thunder=0.9)) == 1   # worse: new alert
    assert Alert.objects.filter(acknowledged=False, level="RED").count() == 1


def test_expired_alert_allows_new_one(airfield):
    evaluate_prediction(make_pred(airfield, thunder=0.6))
    Alert.objects.update(valid_until=timezone.now() - timedelta(minutes=1))
    assert evaluate_prediction(make_pred(airfield, thunder=0.6)) == 1
    assert Alert.objects.count() == 2


def test_longer_horizon_extends_validity(airfield):
    evaluate_prediction(make_pred(airfield, thunder=0.6, horizon=60))
    first = Alert.objects.get().valid_until
    evaluate_prediction(make_pred(airfield, thunder=0.6, horizon=180))
    assert Alert.objects.get().valid_until > first


def test_generate_alerts_uses_latest_per_horizon_and_skips_stale(airfield):
    make_pred(airfield, thunder=0.9, age_min=300)            # stale: ignored
    assert generate_alerts() == 0
    make_pred(airfield, thunder=0.9, age_min=30)             # superseded by the next one
    make_pred(airfield, thunder=0.1)                         # newest 60-min prediction is GREEN
    assert generate_alerts() == 0
    make_pred(airfield, wind=0.55, horizon=180)
    assert generate_alerts() == 1
    assert Alert.objects.get().alert_type == "StrongWind"
