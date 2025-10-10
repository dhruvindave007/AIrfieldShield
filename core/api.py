from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.models import Airfield, Prediction
from alerts.models import Alert
import random, datetime

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
        preds = list(Prediction.objects.filter(airfield=airfield).order_by("-created_at")[:5].values())

        # alerts
        alerts = list(Alert.objects.filter(airfield=airfield, acknowledged=False).order_by("-created_at").values())

        # simulate current weather
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
            {"name": "Satellite Image", "status": "ok"},
            {"name": "RadarFeed", "status": "ok"},
            {"name": "PredictionEngine", "status": "ok"},
        ]
                # --- add trend (last 6h predictions) ---
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=6)
        history_preds = Prediction.objects.filter(
            airfield=airfield, created_at__gte=cutoff
        ).order_by("created_at")

        trend = [
            {
                "time": p.created_at.isoformat(),
                "thunder": p.thunderstorm_prob or 0,
                "gale": p.gale_wind_prob or 0,
            }
            for p in history_preds
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
            "trend": trend,
        })


class PredictionHistoryAPIView(APIView):
    def get(self, request, *args, **kwargs):
        q = request.GET.get("airfield")
        hours = int(request.GET.get("hours", 6))

        airfield = Airfield.objects.filter(icao__iexact=q).first() \
                   or Airfield.objects.filter(name__iexact=q).first()
        if not airfield:
            return Response({"error": f"Airfield {q} not found"}, status=status.HTTP_404_NOT_FOUND)

        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        preds = Prediction.objects.filter(airfield=airfield, created_at__gte=cutoff).order_by("created_at")

        history = [
            {
                "created_at": p.created_at,
                "thunderstorm_prob": p.thunderstorm_prob,
                "gale_wind_prob": p.gale_wind_prob,
            }
            for p in preds
        ]

        return Response({"airfield": airfield.icao, "history": history})
