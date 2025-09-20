# ai_models/tasks.py
from celery import shared_task
import logging

logger = logging.getLogger(__name__)

@shared_task(ignore_result=True)
def run_prediction_cycle():
    logger.info("Starting scheduled prediction cycle (stub).")
    # TODO: call your management command or a service function that:
    # 1) fetches latest observations
    # 2) runs models to produce Prediction objects
    # 3) triggers alert evaluation
    return "ok"
