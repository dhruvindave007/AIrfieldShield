from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.models import Airfield, Prediction
from alerts.models import Alert
from django.utils import timezone
import random

class DashboardFrontendAPIView(APIView):
    def get(self, request, *args, **kwargs):
        q = request.GET.get("airfield")
        if not q:
            return Response({"error": "Missing ?airfield="}, status=status.HTTP_400_BAD_REQUEST)

        airfield = Airfield.objects.filter(icao__iexact=q).first() \
                   or Airfield.objects.filter(name__iexact=q).first()
        if not airfield:
            return Response({"error": f"Airfield {q} not found"}, status=status.HTTP_404_NOT_FOUND)

        # latest predictions
        preds = list(Prediction.objects.filter(airfield=airfield).order_by("-created_at")[:10].values())

        # alerts
        alerts = list(Alert.objects.filter(airfield=airfield, acknowledged=False).order_by("-created_at").values())

        # simulate current weather (kept for demo - real ingestion should replace this)
        weather = {
            "temperature": round(random.uniform(25, 35), 2),
            "humidity": round(random.uniform(50, 95), 2),
            "pressure": round(random.uniform(995, 1015), 2),
            "windSpeed": round(random.uniform(0, 20), 2),
            "windDirection": round(random.uniform(0, 360), 1),
            "radarIntensity": round(random.uniform(0, 1), 2),
        }

        # simulate storm cells (GeoJSON)
        storm_cells = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            airfield.longitude + random.uniform(-0.3, 0.3),
                            airfield.latitude + random.uniform(-0.3, 0.3)
                        ]
                    },
                    "properties": {
                        "type": random.choice(["Thunderstorm", "GaleWind", "RainShower"]),
                        "intensity": round(random.uniform(0.3, 0.9), 2),
                        "movement": {
                            "direction": random.randint(0, 360),
                            "speed_kmh": random.randint(5, 40),
                        },
                    },
                }
                for _ in range(3)
            ],
        }

        # mark data sources as OK
        sources = [
            {"name": "Satellite", "status": "ok"},
            {"name": "RadarFeed", "status": "ok"},
            {"name": "PredictionEngine", "status": "ok"},
        ]

        return Response({
            "airfield": {
                "id": airfield.id,
                "name": airfield.name,
                "icao": airfield.icao,
                "latitude": airfield.latitude,
                "longitude": airfield.longitude,
            },
            "alerts": alerts,
            "predictions": preds,
            "weatherData": weather,
            "stormCells": storm_cells,
            "dataSources": sources,
        })


class PredictionHistoryAPIView(APIView):
    def get(self, request, *args, **kwargs):
        q = request.GET.get("airfield")
        hours = int(request.GET.get("hours", 6))

        if not q:
            return Response({"error": "Missing ?airfield="}, status=status.HTTP_400_BAD_REQUEST)

        airfield = (
            Airfield.objects.filter(icao__iexact=q).first()
            or Airfield.objects.filter(name__iexact=q).first()
        )
        if not airfield:
            return Response({"error": f"Airfield {q} not found"}, status=status.HTTP_404_NOT_FOUND)

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        preds = (
            Prediction.objects.filter(airfield=airfield, created_at__gte=cutoff)
            .order_by("created_at")
            .values("created_at", "thunderstorm_prob", "gale_wind_prob")
        )

        return Response({"airfield": airfield.icao, "history": list(preds)})