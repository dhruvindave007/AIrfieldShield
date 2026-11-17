"""Turn predictions into de-duplicated, escalating alerts.

Alert levels are not hand-picked probability cut-offs: each prediction carries
the thresholds that reached 25% / 50% / 75% precision on the validation year
(YELLOW / ORANGE / RED). Settings provide a fallback for predictions without
model-derived thresholds.
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from ai_models.ml.model import level_for
from core.models import Airfield, Prediction

from .models import Alert

LEVEL_ORDER = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
HAZARDS = {"Thunderstorm": "thunderstorm_prob", "StrongWind": "gale_wind_prob"}
HAZARD_LABEL = {"Thunderstorm": "Thunderstorm", "StrongWind": "Strong wind (>= 25 kt)"}


def thresholds_for(pred: Prediction, hazard: str) -> dict:
    model_thresholds = ((pred.details or {}).get("thresholds") or {}).get(hazard)
    return model_thresholds or settings.AIRFIELDSHIELD["FALLBACK_THRESHOLDS"]


def level_for_prediction(pred: Prediction, hazard: str) -> str:
    return level_for(getattr(pred, HAZARDS[hazard]), thresholds_for(pred, hazard))


def evaluate_prediction(pred: Prediction, now=None) -> int:
    """Create or escalate alerts for one prediction. Returns the number of
    alerts created or escalated.

    Rules, per (airfield, hazard):
    * GREEN never creates an alert; existing alerts simply expire at valid_until.
    * A still-valid alert of the same or higher level is not duplicated; if it
      is unacknowledged its validity is extended.
    * A higher level escalates an unacknowledged alert in place, or creates a
      new alert if the previous one was already acknowledged.
    """
    now = now or timezone.now()
    changed = 0
    for hazard, field in HAZARDS.items():
        prob = getattr(pred, field)
        level = level_for_prediction(pred, hazard)
        if level == "GREEN":
            continue
        valid_until = pred.created_at + timedelta(minutes=pred.horizon_minutes)
        if valid_until < now:
            continue  # stale prediction
        message = (f"{HAZARD_LABEL[hazard]}: {prob:.0%} probability within "
                   f"{pred.horizon_minutes} min")
        meta = {"prediction_id": pred.id, "probability": prob,
                "horizon_minutes": pred.horizon_minutes, "model_version": pred.model_version}
        current = (Alert.objects
                   .filter(airfield=pred.airfield, alert_type=hazard)
                   .filter(Q(valid_until__isnull=True) | Q(valid_until__gte=now))
                   .order_by("-created_at").first())
        if current is None:
            Alert.objects.create(airfield=pred.airfield, alert_type=hazard, level=level,
                                 message=message, valid_until=valid_until, meta=meta)
            changed += 1
        elif LEVEL_ORDER[level] > LEVEL_ORDER.get(current.level, 0):
            if current.acknowledged:
                Alert.objects.create(airfield=pred.airfield, alert_type=hazard, level=level,
                                     message=message, valid_until=valid_until, meta=meta)
            else:
                current.level, current.message, current.meta = level, message, meta
                current.valid_until = max(current.valid_until or valid_until, valid_until)
                current.save(update_fields=["level", "message", "meta", "valid_until"])
            changed += 1
        elif not current.acknowledged and current.valid_until and valid_until > current.valid_until:
            current.valid_until = valid_until
            current.save(update_fields=["valid_until"])
    return changed


def generate_alerts(now=None, max_age_minutes: int = 120) -> int:
    """Evaluate the newest prediction of every horizon for every airfield."""
    now = now or timezone.now()
    cutoff = now - timedelta(minutes=max_age_minutes)
    changed = 0
    for airfield in Airfield.objects.all():
        seen = set()
        recent = airfield.predictions.filter(created_at__gte=cutoff).order_by("-created_at")
        for pred in recent:
            if pred.horizon_minutes in seen:
                continue
            seen.add(pred.horizon_minutes)
            changed += evaluate_prediction(pred, now=now)
    return changed
