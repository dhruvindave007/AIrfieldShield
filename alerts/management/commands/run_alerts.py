# alerts/management/commands/run_alerts.py
from django.core.management.base import BaseCommand
from alerts.models import Alert
from core.models import Prediction
from django.utils import timezone
from datetime import timedelta

class Command(BaseCommand):
    help = "Create alerts from recent predictions"

    def handle(self, *args, **options):
        now = timezone.now()
        recent_preds = Prediction.objects.filter(created_at__gte=now - timedelta(hours=2)).order_by('-created_at')
        created = 0
        for p in recent_preds:
            thunder = getattr(p, "thunderstorm_prob", 0) or 0
            gale = getattr(p, "gale_wind_prob", 0) or 0
            # choose alarm if either crosses threshold
            if thunder >= 0.5 or gale >= 0.5:
                level = "RED" if thunder >= 0.75 or gale >= 0.75 else "ORANGE"
                msg_parts = []
                if thunder >= 0.5: msg_parts.append(f"Thunder: {int(thunder*100)}%")
                if gale >= 0.5: msg_parts.append(f"Gale: {int(gale*100)}%")
                Alert.objects.create(
                    airfield=p.airfield,
                    alert_type="Thunderstorm" if thunder>=gale else "GaleWind",
                    message="; ".join(msg_parts),
                    level=level,
                    created_at=now,
                    valid_until=now + timedelta(hours=2),
                )
                created += 1
        self.stdout.write(self.style.SUCCESS(f"Created {created} alerts"))
