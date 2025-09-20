from django.urls import path, include
from rest_framework import routers
from .views import AirfieldViewSet, WeatherObservationViewSet, PredictionViewSet
from . import api

router = routers.DefaultRouter()
router.register(r'airfields', AirfieldViewSet)
router.register(r'observations', WeatherObservationViewSet)
router.register(r'predictions', PredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),  # /api/airfields/, etc.
    path('frontend/dashboard/', api.DashboardFrontendAPIView.as_view(), name='api_frontend_dashboard'),
    path('predictions/history/', api.PredictionHistoryAPIView.as_view(), name='api_prediction_history'),
]
