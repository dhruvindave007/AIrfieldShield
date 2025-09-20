import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airfieldshield.settings")

app = Celery("airfieldshield")
# Prefer configuration from Django settings, using a CELERY namespace
app.config_from_object("django.conf:settings", namespace="CELERY")

# Autodiscover tasks in installed apps under tasks.py
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
