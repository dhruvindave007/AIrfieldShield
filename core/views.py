from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from alerts.models import Alert
from alerts.serializers import AlertSerializer

from .models import Airfield, Prediction, WeatherObservation
from .query import find_airfield, int_param
from .serializers import AirfieldSerializer, PredictionSerializer, WeatherObservationSerializer


class AirfieldViewSet(viewsets.ModelViewSet):
    queryset = Airfield.objects.all().order_by("id")
    serializer_class = AirfieldSerializer


class WeatherObservationViewSet(viewsets.ReadOnlyModelViewSet):
    """Observations are written by the ingestion service only."""
    queryset = WeatherObservation.objects.select_related("station").order_by("-timestamp")
    serializer_class = WeatherObservationSerializer


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """Predictions are written by the prediction service only."""
    queryset = Prediction.objects.select_related("airfield").order_by("-created_at")
    serializer_class = PredictionSerializer


def _airfield_or_error(request):
    query = request.query_params.get("airfield")
    if not query:
        return None, Response({"error": "Missing ?airfield="}, status=status.HTTP_400_BAD_REQUEST)
    airfield = find_airfield(Airfield.objects.all(), query)
    if airfield is None:
        return None, Response({"error": f"Airfield {query} not found"},
                              status=status.HTTP_404_NOT_FOUND)
    return airfield, None


class DashboardFrontendAPIView(APIView):
    """GET /api/frontend/dashboard/?airfield=<id|icao|name>

    Everything returned here comes from the database (real observations, model
    predictions, alerts). Nothing is simulated."""

    def get(self, request):
        airfield, error = _airfield_or_error(request)
        if error:
            return error
        now = timezone.now()

        preds = Prediction.objects.filter(airfield=airfield).select_related("airfield") \
            .order_by("-created_at")[:10]
        alerts = Alert.objects.filter(airfield=airfield, acknowledged=False) \
            .filter(Q(valid_until__isnull=True) | Q(valid_until__gte=now)) \
            .select_related("airfield").order_by("-created_at")

        obs = WeatherObservation.objects.filter(station__airfield=airfield) \
            .order_by("-timestamp").first()
        weather = {}
        if obs:
            weather = {
                "observedAt": obs.timestamp,
                "temperature": obs.temperature_c,
                "humidity": obs.humidity,
                "pressure": obs.pressure_hpa,
                "windSpeed": obs.wind_speed_ms,
                "windGust": obs.wind_gust_ms,
                "windDirection": obs.wind_dir_deg,
                "thunderstorm": obs.thunderstorm,
                "metar": (obs.raw_payload or {}).get("metar"),
            }
            weather = {k: v for k, v in weather.items() if v is not None}

        max_age = timedelta(minutes=settings.AIRFIELDSHIELD["MAX_OBSERVATION_AGE_MIN"])
        latest_pred = preds[0] if preds else None
        sources = [
            {"name": "METAR observations",
             "status": "OK" if obs and now - obs.timestamp <= max_age
             else ("DEGRADED" if obs else "DOWN")},
            {"name": f"Nowcast model ({latest_pred.model_version or 'n/a'})" if latest_pred
             else "Nowcast model",
             "status": "OK" if latest_pred and now - latest_pred.created_at <= max_age
             else ("DEGRADED" if latest_pred else "DOWN")},
            {"name": "Weather radar", "status": "NOT CONNECTED"},
        ]

        return Response({
            "airfield": AirfieldSerializer(airfield).data,
            "alerts": AlertSerializer(alerts, many=True).data,
            "predictions": PredictionSerializer(preds, many=True).data,
            "weatherData": weather,
            "stormCells": None,  # no radar source is connected; nothing is simulated
            "dataSources": sources,
        })


class PredictionHistoryAPIView(APIView):
    """GET /api/predictions/history/?airfield=<id|icao|name>&hours=6&horizon=60"""

    def get(self, request):
        airfield, error = _airfield_or_error(request)
        if error:
            return error
        hours = int_param(request.query_params.get("hours"), default=6, lo=1, hi=168)
        horizon = int_param(request.query_params.get("horizon"), default=60, lo=1, hi=1440)
        since = timezone.now() - timedelta(hours=hours)
        history = (Prediction.objects
                   .filter(airfield=airfield, created_at__gte=since, horizon_minutes=horizon)
                   .order_by("created_at")
                   .values("created_at", "horizon_minutes", "thunderstorm_prob", "gale_wind_prob"))
        return Response({"airfield": airfield.icao, "horizon_minutes": horizon,
                         "history": list(history)})
