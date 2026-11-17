from django.core.management.base import BaseCommand

from ingestion.services import ingest_all


class Command(BaseCommand):
    help = "Ingest recent real METAR observations (public IEM archive, no API key)."

    def add_arguments(self, parser):
        parser.add_argument("--hours", type=int, default=30, help="look-back window (default 30)")
        parser.add_argument("--airfield", type=str, help="only this ICAO code")

    def handle(self, *args, **options):
        result = ingest_all(hours=options["hours"], icao=options.get("airfield"))
        if not result:
            self.stdout.write(self.style.WARNING("No airfields with an ICAO code. Run seed_airfields first."))
        for icao, n in result.items():
            if n < 0:
                self.stdout.write(self.style.ERROR(f"{icao}: ingestion failed (see log)"))
            else:
                self.stdout.write(self.style.SUCCESS(f"{icao}: {n} new observations"))
