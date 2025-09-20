# scripts/seed_demo_data.py
# Run from project root with: python scripts/seed_demo_data.py

import os
import random
from datetime import timedelta

# Point to your Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airfieldshield.settings")

import django
django.setup()

from django.utils import timezone
from core.models import Airfield, WeatherStation, WeatherObservation

def seed_demo():
    now = timezone.now()

    # Create or get an Airfield
    af, created = Airfield.objects.get_or_create(
        icao="TEST",
        defaults={
            "name": "Test Field",
            "latitude": 23.020,    # Ahmedabad-ish example
            "longitude": 72.570,
            "elevation_m": 50.0,
        },
    )
    print("Airfield:", af, "created?" , created)

    # Create or get a WeatherStation linked to the airfield
    ws, ws_created = WeatherStation.objects.get_or_create(
        airfield=af,
        station_id="S1",
        defaults={"latitude": af.latitude, "longitude": af.longitude},
    )
    print("WeatherStation:", ws, "created?", ws_created)

    # Insert observations for the last ~60 minutes (every 5 minutes)
    n_points = 13  # ~60 minutes / 5
    created_count = 0
    for i in range(n_points):
        ts = now - timedelta(minutes=5 * i)
        # create reasonable synthetic weather values with some variability
        temp = 25.0 + random.uniform(-3.0, 5.0)         # deg C
        hum = 60.0 + random.uniform(-15.0, 10.0)        # %
        # create some pressure drop tendency in some points
        pressure = 1006.0 + random.uniform(-8.0, 3.0)   # hPa
        wind_speed = max(0.0, random.uniform(0.0, 12.0))# m/s
        wind_gust = wind_speed + random.uniform(0.0, 8.0)
        wind_dir = random.uniform(0.0, 360.0)

        WeatherObservation.objects.create(
            station=ws,
            timestamp=ts,
            temperature_c=round(temp, 2),
            humidity=round(hum, 2),
            pressure_hpa=round(pressure, 2),
            wind_speed_ms=round(wind_speed, 2),
            wind_gust_ms=round(wind_gust, 2),
            wind_dir_deg=round(wind_dir, 1),
        )
        created_count += 1

    total_obs = WeatherObservation.objects.filter(station__airfield=af).count()
    print(f"Inserted {created_count} observations. Total observations for {af.name}: {total_obs}")

if __name__ == "__main__":
    seed_demo()
