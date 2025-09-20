# AirfieldShield - Django AI/ML Prediction for Thunderstorms & Gale Winds

This repository skeleton sets up a Django project with apps:
- core      : models and API for airfields, stations, observations, predictions
- ingestion : data ingestion pipelines (radar, satellite, AWS, lidar, etc.)
- ai_models : model metadata and training/prediction hooks
- alerts    : alert engine and storage
- dashboard : UI and frontend components
- utils     : shared helpers and utils

Next steps:
1. Add the apps to INSTALLED_APPS in settings.py:
   INSTALLED_APPS += ["core","ingestion","ai_models","alerts","dashboard","utils","rest_framework"]

2. Configure database settings and run:
   python manage.py makemigrations
   python manage.py migrate

3. Implement ingestion and model training code. Use management commands created as stubs:
   - python manage.py ingest_data --source=radar
   - python manage.py predict
   - python manage.py run_alerts

4. Add authentication, thresholds, and a frontend for the dashboard.

