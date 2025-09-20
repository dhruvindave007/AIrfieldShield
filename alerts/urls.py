# alerts/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("latest/", views.LatestAlertsAPIView.as_view(), name="latest_alerts"),
    path("latest-predictions/", views.LatestPredictionsPerAirfield.as_view(), name="latest_predictions"),
    path("ack/<int:pk>/", views.acknowledge_alert, name="acknowledge_alert"),
]
