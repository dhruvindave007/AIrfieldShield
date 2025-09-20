"""
Command: run_alerts
Evaluates recent Prediction objects and raises/stores Alert objects.

Usage:
  python manage.py run_alerts
  python manage.py run_alerts --since-minutes 60
  python manage.py run_alerts --airfield TEST
"""

import logging
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings

from core.models import Airfield
from alerts.models import Alert
from core.models import Prediction

logger = logging.getLogger(__name__)


def prob_to_level(prob: float, base_threshold: float):
    """
    Map probability -> level string.
    - RED for very high (>0.85)
    - ORANGE for high (>=0.6)
    - YELLOW for moderate (>= base_threshold)
    - GREEN otherwise
    """
    if prob is None:
        return "GREEN"
    try:
        p = float(prob)
    except Exception:
        return "GREEN"

    if p >= 0.85:
        return "RED"
    if p >= 0.6:
        return "ORANGE"
    if p >= base_threshold:
        return "YELLOW"
    return "GREEN"


def notify_external(alert: Alert):
    """
    Place-holder for external notifications (email, webhook, SMS).
    Right now just logs. Extend this to call webhooks / push notifications.
    """
    logger.info("NOTIFY: %s | %s | valid_until=%s", alert.airfield, alert.message, alert.valid_until)


class Command(BaseCommand):
    help = "Evaluate predictions and generate Alert objects (one per airfield & hazard)"

    def add_arguments(self, parser):
        parser.add_argument("--since-minutes", type=int, default=180, help="Look for predictions created in this window (minutes)")
        parser.add_argument("--airfield", type=str, help="Filter by airfield ICAO or id (optional)")
        parser.add_argument("--dry-run", action="store_true", help="Do not create Alert rows; only print what would happen")

    def handle(self, *args, **options):
        since_minutes = options.get("since_minutes", 180)
        af_arg = options.get("airfield")
        dry_run = options.get("dry_run", False)

        thunder_threshold = settings.AIRFIELDSHIELD.get("DEFAULT_THUNDERSTORM_THRESHOLD", 0.5)
        gale_prob_threshold = 0.5  # default for gale probability (model-specific)
        now = timezone.now()
        cutoff = now - timedelta(minutes=since_minutes)

        # select airfields
        if af_arg:
            try:
                aid = int(af_arg)
                airfields = Airfield.objects.filter(id=aid)
            except Exception:
                airfields = Airfield.objects.filter(icao__iexact=af_arg) | Airfield.objects.filter(name__iexact=af_arg)
        else:
            airfields = Airfield.objects.all()

        if not airfields.exists():
            self.stdout.write(self.style.WARNING("No airfields found for alert evaluation."))
            return

        created_alerts = 0
        for af in airfields:
            # get latest prediction for this airfield within window
            pred = Prediction.objects.filter(airfield=af, created_at__gte=cutoff).order_by("-created_at").first()
            if not pred:
                logger.debug("No prediction for airfield %s in last %d minutes", af, since_minutes)
                continue

            # Evaluate thunderstorm probability
            tprob = pred.thunderstorm_prob or 0.0
            tlevel = prob_to_level(tprob, thunder_threshold)
            t_msg = f"Thunderstorm risk: {tprob:.2f} in next {pred.horizon_minutes} minutes"
            t_meta = {"prediction_id": pred.id, "thunder_prob": tprob, "prediction_created": pred.created_at.isoformat()}

            # Evaluate gale wind probability
            gprob = pred.gale_wind_prob or 0.0
            glevel = prob_to_level(gprob, gale_prob_threshold)
            g_msg = f"Gale wind risk: {gprob:.2f} in next {pred.horizon_minutes} minutes"
            g_meta = {"prediction_id": pred.id, "gale_prob": gprob, "prediction_created": pred.created_at.isoformat()}

            # Decide whether to raise alerts (only for non-GREEN)
            hazards = [
                ("Thunderstorm", tlevel, t_msg, t_meta),
                ("GaleWind", glevel, g_msg, g_meta),
            ]

            for atype, level, message, meta in hazards:
                if level == "GREEN":
                    # Optionally we could de-ack existing alerts here (not implemented)
                    continue

                valid_until = pred.created_at + timedelta(minutes=pred.horizon_minutes)
                if dry_run:
                    self.stdout.write(f"[DRY] Would create Alert for {af} — {atype} {level}: {message} valid_until {valid_until}")
                    continue

                alert = Alert.objects.create(
                    airfield=af,
                    alert_type=atype,
                    message=message,
                    level=level,
                    valid_until=valid_until,
                    meta=meta,
                )
                created_alerts += 1
                self.stdout.write(self.style.SUCCESS(f"Created Alert {alert.id}: {af} — {atype} {level}"))
                # notify external systems (currently logs)
                notify_external(alert)

        self.stdout.write(self.style.SUCCESS(f"Done. Created {created_alerts} alerts."))
