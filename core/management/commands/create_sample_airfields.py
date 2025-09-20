from django.core.management.base import BaseCommand
from core.models import Airfield

class Command(BaseCommand):
    help = "Create some sample airfields for simulation/demo"

    def handle(self, *args, **options):
        airfields = [
            {"name": "Test Field", "icao": "TEST", "latitude": 23.02, "longitude": 72.57},
            {"name": "Dev Field", "icao": "DVFL", "latitude": 22.50, "longitude": 73.20},
            {"name": "Training Base", "icao": "TRNB", "latitude": 21.15, "longitude": 72.75},
        ]

        for data in airfields:
            obj, created = Airfield.objects.get_or_create(
                icao=data["icao"],
                defaults={
                    "name": data["name"],
                    "latitude": data["latitude"],
                    "longitude": data["longitude"],
                },
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created airfield: {obj.name} ({obj.icao})"))
            else:
                self.stdout.write(self.style.WARNING(f"Airfield already exists: {obj.name} ({obj.icao})"))

        self.stdout.write(self.style.SUCCESS("Sample airfields setup complete."))
