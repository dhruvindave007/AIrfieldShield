#!/usr/bin/env python3
"""
Management command: seed_observations

Creates synthetic WeatherObservation rows for testing.

Usage examples (from project root):
  # Basic: seed for an existing airfield by ID
  python manage.py seed_observations --airfield 1

  # Create a new airfield then seed 2 stations for 3 hours at 1-min freq
  python manage.py seed_observations --create-airfield --name "Testfield" --icao "TEST" --lat 23.03 --lng 72.58

  # Inject a storm period in the middle to test alerts
  python manage.py seed_observations --airfield 1 --inject-storm

Options:
  --airfield: airfield id (int) or ICAO code (string). If omitted and --create-airfield not used, will attempt to use the first Airfield.
  --create-airfield: if set, will create an airfield (requires --name, --icao, --lat, --lng optionally)
  --name, --icao, --lat, --lng: used with --create-airfield
  --n-stations: number of stations to create (default 2)
  --duration-minutes: total minutes of data to create (default 180)
  --freq-seconds: frequency of observations in seconds (default 60)
  --start-offset-minutes: how many minutes in the past to start (default: duration)
  --inject-storm: include a short high-impact storm in the middle
  --seed: RNG seed for reproducibility
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction

import random
import math
from datetime import timedelta
from itertools import islice

# Import models
from core.models import Airfield, WeatherStation, WeatherObservation


def frange(start, stop, step):
    # inclusive range generator for floats
    x = start
    while x <= stop:
        yield x
        x += step


class Command(BaseCommand):
    help = "Seed synthetic WeatherObservation rows for testing the prediction pipeline"

    def add_arguments(self, parser):
        parser.add_argument("--airfield", type=str, help="Airfield id (int) or ICAO code (str). If omitted, use first airfield or create one.")
        parser.add_argument("--create-airfield", action="store_true", help="Create an airfield before seeding (requires --name, --icao, --lat, --lng)")
        parser.add_argument("--name", type=str, help="Airfield name when creating")
        parser.add_argument("--icao", type=str, help="Airfield ICAO when creating")
        parser.add_argument("--lat", type=float, help="Latitude when creating")
        parser.add_argument("--lng", type=float, help="Longitude when creating")
        parser.add_argument("--n-stations", type=int, default=2, help="Number of weather stations to create (default 2)")
        parser.add_argument("--duration-minutes", type=int, default=180, help="Total minutes of data (default 180)")
        parser.add_argument("--freq-seconds", type=int, default=60, help="Observation frequency in seconds (default 60)")
        parser.add_argument("--start-offset-minutes", type=int, default=None, help="How many minutes in past to start (default = duration)")
        parser.add_argument("--inject-storm", action="store_true", help="Inject a short storm period in the middle")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data")

    def handle(self, *args, **options):
        seed = int(options.get("seed") or 42)
        random.seed(seed)

        create_airfield = options.get("create_airfield", False)
        airfield_arg = options.get("airfield")
        name = options.get("name") or "Synthetic Airfield"
        icao = options.get("icao") or f"SYN{random.randint(10,99)}"
        lat = options.get("lat") or 23.02
        lng = options.get("lng") or 72.57

        n_stations = int(options.get("n_stations") or 2)
        duration_minutes = int(options.get("duration_minutes") or 180)
        freq_seconds = int(options.get("freq_seconds") or 60)
        start_offset = options.get("start_offset_minutes")
        inject_storm = options.get("inject_storm", False)

        if start_offset is None:
            start_offset = duration_minutes

        # Find or create airfield
        airfield = None
        if create_airfield:
            airfield = Airfield.objects.create(name=name, icao=icao, latitude=lat, longitude=lng)
            self.stdout.write(self.style.SUCCESS(f"Created Airfield: {airfield}"))
        else:
            if airfield_arg:
                try:
                    aid = int(airfield_arg)
                    airfield = Airfield.objects.filter(id=aid).first()
                except Exception:
                    airfield = Airfield.objects.filter(icao__iexact=airfield_arg).first() or Airfield.objects.filter(name__iexact=airfield_arg).first()

            if not airfield:
                airfield = Airfield.objects.first()

            if not airfield:
                self.stdout.write(self.style.ERROR("No airfield found. Either create one with --create-airfield or ensure one exists."))
                return
            else:
                self.stdout.write(self.style.NOTICE(f"Using Airfield: {airfield}"))

        # Create weather stations if none exist for this airfield
        stations = list(airfield.stations.all())
        created_stations = []
        for i in range(n_stations - len(stations)):
            sid = f"{airfield.icao or airfield.name[:3].upper()}_S{i+1}"
            s = WeatherStation.objects.create(airfield=airfield, station_id=sid, latitude=airfield.latitude + (i * 0.001), longitude=airfield.longitude + (i * 0.001))
            created_stations.append(s)
            stations.append(s)

        if created_stations:
            self.stdout.write(self.style.SUCCESS(f"Created {len(created_stations)} station(s): {[s.station_id for s in created_stations]}"))

        # Generate timestamps list
        now = timezone.now()
        start_time = now - timedelta(minutes=start_offset)
        num_points = int(math.ceil((duration_minutes * 60) / float(freq_seconds)))
        timestamps = [start_time + timedelta(seconds=i * freq_seconds) for i in range(num_points)]

        # Synthetic base climatology derived from airfield lat/lng (small randomization)
        base_temp_c = 25.0 + (random.random() - 0.5) * 6.0  # 22 ± 3 to 28 ±3
        base_humidity = 60 + (random.random() - 0.5) * 20.0
        base_pressure = 1010 + (random.random() - 0.5) * 10.0
        base_wind_mean = 3.0 + (random.random() - 0.5) * 3.0

        self.stdout.write(self.style.NOTICE(f"Seeding {num_points} observations per station ({len(stations)} stations) from {start_time} to {timestamps[-1]}"))

        # Prepare bulk lists
        all_obs = []
        batch_size = 1000  # commit in batches for large inserts

        # Storm injection window (if requested)
        storm_window_start_idx = None
        storm_window_end_idx = None
        if inject_storm:
            # place storm roughly in middle 10% of the window (e.g., 10% of duration)
            mid = int(num_points // 2)
            width = max(3, int(num_points * 0.08))  # at least 3 points, else ~8%
            storm_window_start_idx = max(0, mid - width // 2)
            storm_window_end_idx = min(num_points - 1, mid + width // 2)
            self.stdout.write(self.style.WARNING(f"Injecting storm between idx {storm_window_start_idx} and {storm_window_end_idx}"))

        for s in stations:
            # per-station small offsets to simulate microclimate
            st_temp_offset = (random.random() - 0.5) * 2.0
            st_hum_offset = (random.random() - 0.5) * 6.0
            st_pres_offset = (random.random() - 0.5) * 2.0
            st_wind_offset = (random.random() - 0.5) * 1.5

            for idx, t in enumerate(timestamps):
                # base diurnal variation: simple sine wave over the window
                # normalize idx->0..2pi across the window
                phase = 2.0 * math.pi * (idx / float(max(1, num_points)))
                temp = base_temp_c + st_temp_offset + 2.0 * math.sin(phase) + random.gauss(0, 0.4)
                humidity = min(max(10.0, base_humidity + st_hum_offset + 8.0 * math.cos(phase) + random.gauss(0, 2.0)), 100.0)
                pressure = base_pressure + st_pres_offset + 0.5 * math.cos(phase) + random.gauss(0, 0.6)
                wind_speed = max(0.0, base_wind_mean + st_wind_offset + 1.5 * math.sin(phase * 2) + random.gauss(0, 0.6))
                # gust is wind_speed + random (rare high gusts)
                gust = wind_speed + max(0.0, random.gauss(0, 2.0))
                wind_dir = random.uniform(0, 360)

                # If this index is within the storm window, amplify effects
                if inject_storm and storm_window_start_idx is not None and storm_window_start_idx <= idx <= storm_window_end_idx:
                    # ramp up: stronger gusts, lower pressure, higher humidity
                    gust += random.uniform(8, 18)  # +8 to +18 m/s
                    pressure -= random.uniform(6, 14)  # drop 6-14 hPa
                    humidity = min(100.0, humidity + random.uniform(5, 15))
                    wind_speed = wind_speed + random.uniform(5, 12)
                    # optionally set wind_dir to a consistent storm direction
                    wind_dir = (wind_dir + 120) % 360

                obs = WeatherObservation(
                    station=s,
                    timestamp=t,
                    temperature_c=round(temp, 2),
                    humidity=round(humidity, 2),
                    pressure_hpa=round(pressure, 2),
                    wind_speed_ms=round(wind_speed, 2),
                    wind_gust_ms=round(gust, 2),
                    wind_dir_deg=round(wind_dir, 1),
                    raw_payload={"seed": seed, "generated": True, "idx": idx}
                )
                all_obs.append(obs)

                # Bulk insert in batches
                if len(all_obs) >= batch_size:
                    WeatherObservation.objects.bulk_create(all_obs)
                    all_obs = []

        # Final flush
        if all_obs:
            WeatherObservation.objects.bulk_create(all_obs)
            all_obs = []

        self.stdout.write(self.style.SUCCESS("Seeding complete."))

        # Summary
        total_inserted = WeatherObservation.objects.filter(station__airfield=airfield).count()
        self.stdout.write(self.style.SUCCESS(f"Total observations in DB for airfield {airfield}: {total_inserted}"))
        if inject_storm:
            # show the storm sample stats
            idx0 = storm_window_start_idx or 0
            idx1 = storm_window_end_idx or 0
            t0 = timestamps[idx0]
            t1 = timestamps[idx1]
            self.stdout.write(self.style.WARNING(f"Storm period (approx): {t0} to {t1}"))
