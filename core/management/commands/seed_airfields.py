# core/management/commands/seed_airfields.py
from django.core.management.base import BaseCommand
from core.models import Airfield

AIRFIELDS = [
    {"name": "Delhi", "icao": "VIDP", "latitude": 28.5562, "longitude": 77.1000},
    {"name": "Mumbai", "icao": "VABB", "latitude": 19.0896, "longitude": 72.8656},
    {"name": "Chennai", "icao": "VOMM", "latitude": 12.9941, "longitude": 80.1709},
    {"name": "Kolkata", "icao": "VECC", "latitude": 22.6547, "longitude": 88.4467},
    {"name": "Bengaluru", "icao": "VOBL", "latitude": 13.1986, "longitude": 77.7066},
    {"name": "Ahmedabad", "icao": "VAAH", "latitude": 23.0772, "longitude": 72.6347},
    {"name": "Test Field", "icao": "TEST", "latitude": 23.02, "longitude": 72.57},
]

class Command(BaseCommand):
    help = "Seed major Indian airfields into DB"

    def handle(self, *args, **options):
        for af in AIRFIELDS:
            obj, created = Airfield.objects.get_or_create(
                icao=af["icao"],
                defaults={
                    "name": af["name"],
                    "latitude": af["latitude"],
                    "longitude": af["longitude"],
                },
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Added {af['name']} ({af['icao']})"))
            else:
                self.stdout.write(self.style.WARNING(f"Already exists {af['name']} ({af['icao']})"))
