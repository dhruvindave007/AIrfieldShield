from django.core.management.base import BaseCommand

from ai_models.ml.stations import STATIONS
from core.models import Airfield, WeatherStation


class Command(BaseCommand):
    help = "Create the airfields the model was trained on, each with its METAR station."

    def handle(self, *args, **options):
        for icao, (name, lat, lon, elev) in STATIONS.items():
            airfield, created = Airfield.objects.update_or_create(
                icao=icao,
                defaults={"name": name, "latitude": lat, "longitude": lon, "elevation_m": elev},
            )
            WeatherStation.objects.get_or_create(
                airfield=airfield, station_id=icao, defaults={"latitude": lat, "longitude": lon})
            self.stdout.write(f"{'Created' if created else 'Updated'} {airfield}")
