# management/commands/create_sample_airfields.py
from django.core.management.base import BaseCommand
from core.models import Airfield

SAMPLES = [
    {"name": "Test Field", "icao": "TEST", "latitude": 23.02, "longitude": 72.57},
    {"name": "Dev Field", "icao": "DVFL", "latitude": 22.5, "longitude": 72.9},
    {"name": "North Airbase", "icao": "NORTH", "latitude": 24.0, "longitude": 73.0},
]

class Command(BaseCommand):
    help = "Create sample airfields for testing"

    def handle(self, *args, **options):
        for s in SAMPLES:
            af, created = Airfield.objects.get_or_create(icao=s["icao"], defaults={
                "name": s["name"], "latitude": s["latitude"], "longitude": s["longitude"]
            })
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created {af}"))
            else:
                self.stdout.write(self.style.NOTICE(f"Airfield {af} already exists"))
