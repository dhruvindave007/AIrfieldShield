# ai_models/services.py
import logging
from datetime import datetime, timedelta, date
from django.utils import timezone
from core.models import WeatherObservation, Airfield, Prediction
from decimal import Decimal

logger = logging.getLogger(__name__)


def sanitize_for_json(obj):
    """
    Same sanitizer used by predict command — convert datetimes, numpy, Decimal, etc., recursively.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    from decimal import Decimal as _Decimal
    from datetime import datetime as _dt, date as _d

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (_dt, _d)):
        return obj.isoformat()
    if isinstance(obj, _Decimal):
        return float(obj)
    if _np is not None and isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None


def get_latest_observations(airfield: Airfield, minutes=30):
    """
    Return latest observations for the given airfield in the past `minutes`.
    """
    cutoff = timezone.now() - timedelta(minutes=minutes)
    stations = airfield.stations.all()
    qs = WeatherObservation.objects.filter(station__in=stations, timestamp__gte=cutoff).order_by("-timestamp")
    return qs


def run_simple_threshold_predictor(airfield: Airfield, horizon_minutes=60):
    """
    A very simple rule-based predictor as a placeholder.
    Replace with LSTM/CNN/ensemble model integration later.
    """
    observations = get_latest_observations(airfield, minutes=60)
    # basic rule: if latest gust > threshold -> high gale prob
    latest = observations.first()
    thunder_prob = 0.0
    gale_prob = 0.0
    reason = []

    if latest:
        if latest.wind_gust_ms and latest.wind_gust_ms >= 15.0:
            gale_prob = 0.9
            reason.append(f"recent gust {latest.wind_gust_ms} m/s")
        if latest.pressure_hpa and latest.pressure_hpa < 1005:
            thunder_prob = min(0.6 + (1006 - latest.pressure_hpa) * 0.05, 0.95)
            reason.append(f"pressure drop to {latest.pressure_hpa} hPa")

    details = {"reasons": reason}
    # include some meta, ensure sanitized
    meta = {"latest_obs_ts": latest.timestamp if latest else None, "n_obs": observations.count()}
    details["meta"] = meta

    # SANITIZE details before saving
    details_safe = sanitize_for_json(details)

    pred = Prediction.objects.create(
        airfield=airfield,
        horizon_minutes=horizon_minutes,
        thunderstorm_prob=thunder_prob,
        gale_wind_prob=gale_prob,
        details=details_safe,
        confidence=max(thunder_prob, gale_prob),
    )
    logger.info(f"Saved prediction: {pred}")
    return pred
