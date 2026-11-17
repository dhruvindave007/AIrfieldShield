"""Real observation ingestion: recent METAR reports from the public IEM archive."""
from __future__ import annotations

import io
import logging
import math
from datetime import timedelta, timezone as dt_timezone

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone

from ai_models.ml.dataset import raw_iem_to_obs
from core.models import Airfield, WeatherObservation, WeatherStation

logger = logging.getLogger(__name__)

FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "vsby", "metar"]


def fetch_recent_metars(icao: str, hours: int = 30, now=None, session=None) -> pd.DataFrame:
    """Download the last ``hours`` of METARs for one station as canonical observations."""
    now = now or timezone.now()
    start = (now - timedelta(hours=hours)).astimezone(dt_timezone.utc)
    end = (now + timedelta(days=1)).astimezone(dt_timezone.utc)
    params = [("station", icao)] + [("data", f) for f in FIELDS] + [
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", 3), ("report_type", 4),
    ]
    http = session or requests
    resp = http.get(settings.AIRFIELDSHIELD["METAR_ARCHIVE_URL"], params=params, timeout=60,
                    headers={"User-Agent": "airfieldshield"})
    resp.raise_for_status()
    return parse_iem_csv(resp.text, since=start)


def parse_iem_csv(text: str, since=None) -> pd.DataFrame:
    """Parse an IEM 'onlycomma' CSV into canonical observations (plus raw 'metar')."""
    if not text.startswith("station,valid"):
        raise ValueError(f"Unexpected METAR archive response: {text[:80]!r}")
    raw = pd.read_csv(io.StringIO(text), na_values=["M"], dtype={"metar": str})
    obs = raw_iem_to_obs(raw)
    if since is not None:
        obs = obs[obs["valid"] >= since]
    return obs


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _flag(v):
    n = _num(v)
    return None if n is None else bool(n)


def store_observations(airfield: Airfield, obs: pd.DataFrame, station_id: str | None = None) -> int:
    """Idempotently upsert canonical observations. Returns the number of new rows."""
    station, _ = WeatherStation.objects.get_or_create(
        airfield=airfield, station_id=station_id or airfield.icao or f"AF{airfield.pk}",
        defaults={"latitude": airfield.latitude or 0.0, "longitude": airfield.longitude or 0.0})
    created = 0
    for row in obs.to_dict("records"):
        metar = row.get("metar")
        _, was_created = WeatherObservation.objects.update_or_create(
            station=station, timestamp=pd.Timestamp(row["valid"]).to_pydatetime(),
            defaults={
                "temperature_c": _num(row.get("temp_c")),
                "dewpoint_c": _num(row.get("dewpoint_c")),
                "humidity": _num(row.get("rh")),
                "pressure_hpa": _num(row.get("pressure_hpa")),
                "wind_speed_ms": _num(row.get("wind_ms")),
                "wind_gust_ms": _num(row.get("gust_ms")),
                "wind_dir_deg": _num(row.get("wind_dir")),
                "visibility_km": _num(row.get("visibility_km")),
                "thunderstorm": _flag(row.get("ts")),
                "cumulonimbus": _flag(row.get("cb")),
                "precipitation": _flag(row.get("precip")),
                "trend_thunderstorm": _flag(row.get("trend_ts")),
                "raw_payload": {"metar": metar} if isinstance(metar, str) else None,
            })
        created += int(was_created)
    return created


def ingest_all(hours: int = 30, icao: str | None = None) -> dict[str, int]:
    """Ingest recent METARs for every airfield with an ICAO code. One failing
    station does not abort the others; failures are reported as -1."""
    airfields = Airfield.objects.exclude(icao__isnull=True).exclude(icao="")
    if icao:
        airfields = airfields.filter(icao__iexact=icao)
    result = {}
    for af in airfields:
        try:
            obs = fetch_recent_metars(af.icao, hours=hours)
            result[af.icao] = store_observations(af, obs)
        except Exception:
            logger.exception("METAR ingestion failed for %s", af.icao)
            result[af.icao] = -1
    return result
