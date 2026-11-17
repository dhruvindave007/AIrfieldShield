from django.core.management.base import BaseCommand, CommandError

from ai_models.services import get_bundle, predict_for_airfield
from core.models import Airfield


def _fmt(p):
    return "n/a" if p is None else f"{p:.3f}"


class Command(BaseCommand):
    help = "Run the trained nowcasting model for every airfield with recent observations."

    def handle(self, *args, **options):
        try:
            bundle = get_bundle()
        except FileNotFoundError as exc:
            raise CommandError(str(exc))
        n = 0
        for airfield in Airfield.objects.all():
            label = airfield.icao or airfield.name
            preds = predict_for_airfield(airfield, bundle=bundle)
            if not preds:
                self.stdout.write(self.style.WARNING(f"{label}: no recent observations, skipped"))
            for p in preds:
                n += 1
                levels = p.details["levels"]
                self.stdout.write(
                    f"{label} +{p.horizon_minutes}min: "
                    f"thunder={_fmt(p.thunderstorm_prob)} [{levels['Thunderstorm']}] "
                    f"strong_wind={_fmt(p.gale_wind_prob)} [{levels['StrongWind']}]")
        self.stdout.write(self.style.SUCCESS(f"Created {n} predictions."))
