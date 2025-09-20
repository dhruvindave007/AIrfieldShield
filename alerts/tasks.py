# alerts/tasks.py
from django.utils import timezone
from django.db.models import Q
from django.conf import settings

from core.models import Airfield, Prediction
from .models import Alert

DEFAULT_THUNDER = settings.AIRFIELDSHIELD.get("DEFAULT_THUNDERSTORM_THRESHOLD", 0.5)
DEFAULT_GALE = settings.AIRFIELDSHIELD.get("DEFAULT_GALE_PROB_THRESHOLD", DEFAULT_THUNDER)

def generate_alerts(force: bool = False):
    """
    Scan latest predictions and create alerts where thresholds exceeded.
    Returns count of new alerts created.
    """
    now = timezone.now()
    created = 0
    for af in Airfield.objects.all():
        latest = af.predictions.order_by("-created_at").first()
        if not latest:
            continue

        t_prob = float(latest.thunderstorm_prob or 0.0)
        g_prob = float(latest.gale_wind_prob or 0.0)

        # Thunder
        if force or t_prob >= DEFAULT_THUNDER:
            exists = Alert.objects.filter(
                airfield=af,
                alert_type="Thunderstorm",
                acknowledged=False
            ).filter(Q(valid_until__isnull=True) | Q(valid_until__gte=now)).exists()
            if not exists:
                level = "RED" if t_prob >= 0.8 else "ORANGE"
                Alert.objects.create(
                    airfield=af,
                    alert_type="Thunderstorm",
                    message=f"Thunderstorm risk {t_prob:.2f}",
                    level=level,
                    valid_until=now + timezone.timedelta(minutes=60),
                    meta={"prediction_id": latest.id},
                )
                created += 1

        # Gale
        if force or g_prob >= DEFAULT_GALE:
            exists = Alert.objects.filter(
                airfield=af,
                alert_type="GaleWind",
                acknowledged=False
            ).filter(Q(valid_until__isnull=True) | Q(valid_until__gte=now)).exists()
            if not exists:
                level = "RED" if g_prob >= 0.8 else "ORANGE"
                Alert.objects.create(
                    airfield=af,
                    alert_type="GaleWind",
                    message=f"Gale wind risk {g_prob:.2f}",
                    level=level,
                    valid_until=now + timezone.timedelta(minutes=60),
                    meta={"prediction_id": latest.id},
                )
                created += 1
    return created
