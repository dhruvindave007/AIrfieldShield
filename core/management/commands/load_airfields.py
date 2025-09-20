from django.core.management.base import BaseCommand
from core.models import Airfield

INDIAN_AIRFIELDS = [
    {"name": "Delhi Indira Gandhi Intl", "icao": "VIDP", "lat": 28.5562, "lon": 77.1000},
    {"name": "Mumbai Chhatrapati Shivaji Intl", "icao": "VABB", "lat": 19.0896, "lon": 72.8656},
    {"name": "Chennai Intl", "icao": "VOMM", "lat": 12.9941, "lon": 80.1709},
    {"name": "Kolkata Netaji Subhash Chandra Bose", "icao": "VECC", "lat": 22.6547, "lon": 88.4467},
    {"name": "Bengaluru Kempegowda Intl", "icao": "VOBL", "lat": 13.1986, "lon": 77.7066},
    {"name": "Ahmedabad Sardar Vallabhbhai Patel Intl", "icao": "VAAH", "lat": 23.0774, "lon": 72.6347},
]

class Command(BaseCommand):
    help = "Load sample Indian airfields with lat/lon"

    def handle(self, *args, **kwargs):
        for af in INDIAN_AIRFIELDS:
            obj, created = Airfield.objects.update_or_create(
                icao=af["icao"],
                defaults={
                    "name": af["name"],
                    "latitude": af["lat"],
                    "longitude": af["lon"],
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created {obj}"))
            else:
                self.stdout.write(self.style.WARNING(f"Updated {obj}"))
