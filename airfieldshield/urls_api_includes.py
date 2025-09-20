# Add this file's contents into your project urls.py or import it there.
from django.urls import path, include

urlpatterns = [
    path('', include('core.urls')),
    path('ingestion/', include('ingestion.urls') if hasattr(__import__('ingestion'), 'urls') else []),
    path('alerts/', include('alerts.urls') if hasattr(__import__('alerts'), 'urls') else []),
]
