from django.core.management.base import BaseCommand
from core.models import Airfield, WeatherStation

class Command(BaseCommand):
    help = "Create sample WeatherStations for each Airfield"

    def handle(self, *args, **options):
        for airfield in Airfield.objects.all():
            ws, created = WeatherStation.objects.get_or_create(
                airfield=airfield,
                defaults={
                    "name": f"{airfield.name} AWS",
                    "latitude": airfield.latitude,
                    "longitude": airfield.longitude,
                },
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created station {ws.name} for {airfield.icao}"))
            else:
                self.stdout.write(self.style.WARNING(f"Station already exists for {airfield.icao}"))
