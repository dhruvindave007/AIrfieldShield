# core/storms_api.py
import random
import math
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.models import Airfield


def generate_storm_cells_for_airfield(af, count=3):
    """
    Generate synthetic storm cells around an airfield.
    Each storm cell is a polygon approximating a circle,
    with intensity and movement metadata.
    """
    base_lat = float(af.latitude or 0.0)
    base_lon = float(af.longitude or 0.0)
    now = timezone.now().isoformat()

    features = []
    for i in range(count):
        # random offset in degrees (approx 100km ~ 1 degree lat)
        offset_lat = random.uniform(-0.3, 0.3)
        offset_lon = random.uniform(-0.3, 0.3)
        center_lat = base_lat + offset_lat
        center_lon = base_lon + offset_lon

        # storm cell size in km → ~ degrees
        radius_km = random.uniform(5, 20)
        radius_deg = radius_km / 111.0  # convert to degrees latitude

        # approximate circle polygon
        points = []
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            lat = center_lat + radius_deg * math.cos(rad)
            lon = center_lon + radius_deg * math.sin(rad) / math.cos(math.radians(center_lat))
            points.append([lon, lat])
        points.append(points[0])  # close polygon

        intensity = round(random.uniform(0.2, 1.0), 2)  # 0.0-1.0 scale
        cell_type = random.choice(["Thunderstorm", "GaleWind", "RainShower"])

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [points]},
            "properties": {
                "id": f"cell-{i}",
                "type": cell_type,
                "intensity": intensity,
                "movement": {
                    "direction": random.randint(0, 360),
                    "speed_kmh": random.randint(10, 40),
                },
                "timestamp": now,
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


class StormCellsAPIView(APIView):
    """
    GET /api/frontend/storms/?airfield=ICAO
    Returns a GeoJSON FeatureCollection of simulated storm cells
    around the requested airfield.
    """

    def get(self, request):
        q = request.query_params.get("airfield")
        af = None
        if q:
            try:
                af = Airfield.objects.filter(id=int(q)).first()
            except Exception:
                af = (
                    Airfield.objects.filter(icao__iexact=q).first()
                    or Airfield.objects.filter(name__icontains=q).first()
                )

        if not af:
            return Response({"detail": "Airfield not found"}, status=status.HTTP_404_NOT_FOUND)

        payload = generate_storm_cells_for_airfield(af)
        return Response(payload)
