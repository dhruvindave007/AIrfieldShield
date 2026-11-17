import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(ignore_result=True)
def run_nowcast_cycle(hours: int = 30):
    """Scheduled cycle (Celery beat): ingest METARs -> predict -> evaluate alerts."""
    from alerts.services import generate_alerts
    from ingestion.services import ingest_all

    from .services import predict_all

    ingested = ingest_all(hours=hours)
    preds = predict_all()
    changed = generate_alerts()
    logger.info("nowcast cycle: ingested=%s predictions=%d alerts_changed=%d",
                ingested, len(preds), changed)
    return {"predictions": len(preds), "alerts_changed": changed}
