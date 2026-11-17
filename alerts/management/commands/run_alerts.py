from django.core.management.base import BaseCommand

from alerts.services import generate_alerts


class Command(BaseCommand):
    help = "Evaluate the latest predictions and create / escalate alerts."

    def add_arguments(self, parser):
        parser.add_argument("--since-minutes", type=int, default=120,
                            help="ignore predictions older than this (default 120)")

    def handle(self, *args, **options):
        changed = generate_alerts(max_age_minutes=options["since_minutes"])
        self.stdout.write(self.style.SUCCESS(f"Alerts created or escalated: {changed}"))
