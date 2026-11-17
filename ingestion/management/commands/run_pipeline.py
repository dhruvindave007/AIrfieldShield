from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run one full cycle: ingest METARs -> predict -> evaluate alerts"

    def add_arguments(self, parser):
        parser.add_argument("--hours", type=int, default=30, help="ingestion look-back window")
        parser.add_argument("--skip-ingest", action="store_true",
                            help="use observations already in the database")

    def handle(self, *args, **options):
        if not options["skip_ingest"]:
            call_command("ingest_metar", hours=options["hours"], stdout=self.stdout)
        call_command("predict", stdout=self.stdout)
        call_command("run_alerts", stdout=self.stdout)
