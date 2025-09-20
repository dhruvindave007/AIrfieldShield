import logging
import subprocess
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run full pipeline: ingest -> predict -> alerts"

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=10,
            help="Number of simulated observations to ingest"
        )
        parser.add_argument(
            "--since-minutes",
            type=int,
            default=120,
            help="How far back to check alerts (minutes)"
        )

    def handle(self, *args, **options):
        count = options["count"]
        since_minutes = options["since_minutes"]

        self.stdout.write(self.style.MIGRATE_HEADING("=== Running Full Pipeline ==="))

        # Step 1: simulate ingestion
        try:
            self.stdout.write(self.style.NOTICE(f"Step 1: Simulating {count} observations..."))
            subprocess.run(
                ["python", "manage.py", "simulate_ingest", f"--count={count}"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Ingestion failed: %s", e)
            self.stdout.write(self.style.ERROR("Ingestion step failed"))

        # Step 2: run predictions
        try:
            self.stdout.write(self.style.NOTICE("Step 2: Running predictions..."))
            subprocess.run(
                ["python", "manage.py", "predict", "--force-sim"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Prediction failed: %s", e)
            self.stdout.write(self.style.ERROR("Prediction step failed"))

        # Step 3: run alerts
        try:
            self.stdout.write(self.style.NOTICE("Step 3: Checking alerts..."))
            subprocess.run(
                ["python", "manage.py", "run_alerts", f"--since-minutes={since_minutes}"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Alerts failed: %s", e)
            self.stdout.write(self.style.ERROR("Alert step failed"))

        self.stdout.write(self.style.SUCCESS("Pipeline completed successfully."))
