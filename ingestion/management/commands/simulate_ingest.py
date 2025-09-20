# ingestion/management/commands/simulate_ingest.py
from django.core.management.base import BaseCommand
from core.models import WeatherObservation, Airfield, WeatherStation
import random, datetime

class Command(BaseCommand):
    help = "Simulate ingest of weather observations"

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=5)

    def handle(self, *args, **options):
        count = options["count"]
        for af in Airfield.objects.all():
            station, _ = WeatherStation.objects.get_or_create(
                station_id=f"{af.icao}_AWS",
                defaults={"airfield": af, "latitude": af.latitude, "longitude": af.longitude}
            )
            for _ in range(count):
                WeatherObservation.objects.create(
                    station=station,
                    timestamp=datetime.datetime.utcnow(),
                    temperature_c=round(random.uniform(20, 36), 2),
                    humidity=round(random.uniform(40, 95), 1),
                    pressure_hpa=round(random.uniform(993, 1015), 2),
                    wind_speed_ms=round(random.uniform(0, 18), 2),
                    wind_gust_ms=round(random.uniform(0, 36), 2),
                    wind_dir_deg=round(random.uniform(0, 360), 1),
                    raw_payload={},
                )
            self.stdout.write(self.style.SUCCESS(f"Simulated {count} obs for {af.icao}"))
