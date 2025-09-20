# core/views.py
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta

from .models import Airfield, WeatherObservation, Prediction
# NEW
from .serializers import AirfieldSerializer, WeatherObservationSerializer, PredictionSerializer

class AirfieldViewSet(viewsets.ModelViewSet):
    queryset = Airfield.objects.all()
    serializer_class = AirfieldSerializer

class WeatherObservationViewSet(viewsets.ModelViewSet):
    queryset = WeatherObservation.objects.all().order_by("-timestamp")

    serializer_class = WeatherObservationSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all().order_by("-created_at")
    serializer_class = PredictionSerializer

class PredictionHistoryAPIView(APIView):
    """
    GET /api/predictions/history/?airfield=<id|icao|name>&hours=6
    """
    def get(self, request):
        airfield_q = request.query_params.get("airfield")
        hours = int(request.query_params.get("hours", 6))
        since = timezone.now() - timedelta(hours=hours)

        qs = Prediction.objects.filter(created_at__gte=since)

        if airfield_q:
            try:
                aid = int(airfield_q)
                qs = qs.filter(airfield__id=aid)
            except Exception:
                qs = qs.filter(airfield__icao__iexact=airfield_q) | qs.filter(airfield__name__icontains=airfield_q)

        qs = qs.order_by("created_at")
        serializer = PredictionSerializer(qs, many=True, context={"request": request})
        return Response({"history": serializer.data}, status=status.HTTP_200_OK)
