from django.core.management.base import BaseCommand
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Ingest data from configured sources (radar, satellite, AWS, historical)"

    def add_arguments(self, parser):
        parser.add_argument('--source', type=str, help='Data source to ingest (radar, satellite, aws, history)', default='all')

    def handle(self, *args, **options):
        source = options['source']
        logger.info(f"Starting ingest for source={source}")
        # TODO: implement actual ingestion pipelines
        self.stdout.write(self.style.SUCCESS(f"Ingested data for source={source} (stub)"))
