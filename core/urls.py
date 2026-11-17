from django.urls import include, path
from rest_framework import routers

from . import views

router = routers.DefaultRouter()
router.register(r"airfields", views.AirfieldViewSet)
router.register(r"observations", views.WeatherObservationViewSet)
router.register(r"predictions", views.PredictionViewSet)

urlpatterns = [
    # Declared before the router so "predictions/history/" is not captured as a pk.
    path("predictions/history/", views.PredictionHistoryAPIView.as_view(), name="api_prediction_history"),
    path("frontend/dashboard/", views.DashboardFrontendAPIView.as_view(), name="api_frontend_dashboard"),
    path("", include(router.urls)),
]
