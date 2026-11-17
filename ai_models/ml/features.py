"""Feature engineering and label construction on the hourly table.

Every feature at row ``h`` is computed from rows <= h only (backward rolling
windows / lags). Labels look strictly forward: (h, h+H]. The same
``build_features`` function is used offline (training) and online (serving).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 34 kt is the WMO gale threshold (Beaufort 8); it is far too rare at these
# airfields to evaluate, so the wind hazard is defined at 25 kt (strong wind).
# See docs/evaluation_report.md for the event counts behind this choice.
STRONG_WIND_KT = 25
STRONG_WIND_MS = STRONG_WIND_KT * 0.514444
HOURS_SINCE_CAP = 72.0

TARGETS = {
    # name: (kind, horizon_hours)
    "thunder_1h": ("thunder", 1),
    "thunder_3h": ("thunder", 3),
    "wind_1h": ("wind", 1),
    "wind_3h": ("wind", 3),
}

FEATURE_COLUMNS = [
    # current state
    "temp_c", "dewpoint_c", "rh", "dewpoint_depression", "pressure_hpa", "visibility_km",
    "wind_ms", "gust_ms", "peak_wind_ms", "wind_u", "wind_v",
    "ts", "cb", "precip", "trend_ts",
    # tendencies
    "pressure_d1h", "pressure_d3h", "temp_d1h", "temp_d3h", "dewpoint_d1h", "dewpoint_d3h",
    "wind_d1h", "visibility_d1h",
    # recent history
    "peak_wind_max3h", "peak_wind_max6h", "ts_sum3h", "ts_sum6h", "ts_sum24h",
    "cb_sum3h", "cb_sum6h", "precip_sum3h", "strong_wind_sum6h",
    "hours_since_ts", "temp_max6h_minus_now",
    # time / place
    "solar_hour_sin", "solar_hour_cos", "doy_sin", "doy_cos",
    "latitude", "longitude", "elevation_m",
    # data quality
    "n_reports",
]


def peak_wind(hourly: pd.DataFrame) -> pd.Series:
    """Max of sustained wind and gust (gusts are only reported when present)."""
    return pd.Series(
        np.fmax(hourly["wind_ms"].astype(float).to_numpy(), hourly["gust_ms"].astype(float).to_numpy()),
        index=hourly.index,
    )


def build_features(hourly: pd.DataFrame, latitude: float, longitude: float,
                   elevation_m: float = 0.0) -> pd.DataFrame:
    """Compute model features for every row of a gap-free hourly table."""
    h = hourly
    f = pd.DataFrame(index=h.index)
    for c in ["temp_c", "dewpoint_c", "rh", "pressure_hpa", "visibility_km",
              "wind_ms", "gust_ms", "ts", "cb", "precip", "trend_ts"]:
        f[c] = h[c].astype(float)
    f["dewpoint_depression"] = f["temp_c"] - f["dewpoint_c"]
    peak = peak_wind(h)
    f["peak_wind_ms"] = peak
    rad = np.deg2rad(h["wind_dir"].astype(float))
    f["wind_u"] = -f["wind_ms"] * np.sin(rad)
    f["wind_v"] = -f["wind_ms"] * np.cos(rad)

    f["pressure_d1h"] = f["pressure_hpa"].diff(1)
    f["pressure_d3h"] = f["pressure_hpa"].diff(3)
    f["temp_d1h"] = f["temp_c"].diff(1)
    f["temp_d3h"] = f["temp_c"].diff(3)
    f["dewpoint_d1h"] = f["dewpoint_c"].diff(1)
    f["dewpoint_d3h"] = f["dewpoint_c"].diff(3)
    f["wind_d1h"] = f["wind_ms"].diff(1)
    f["visibility_d1h"] = f["visibility_km"].diff(1)

    def roll(s, n, how):
        return getattr(s.rolling(n, min_periods=1), how)()

    f["peak_wind_max3h"] = roll(peak, 3, "max")
    f["peak_wind_max6h"] = roll(peak, 6, "max")
    f["ts_sum3h"] = roll(f["ts"], 3, "sum")
    f["ts_sum6h"] = roll(f["ts"], 6, "sum")
    f["ts_sum24h"] = roll(f["ts"], 24, "sum")
    f["cb_sum3h"] = roll(f["cb"], 3, "sum")
    f["cb_sum6h"] = roll(f["cb"], 6, "sum")
    f["precip_sum3h"] = roll(f["precip"], 3, "sum")
    strong = (peak >= STRONG_WIND_MS).astype(float).where(peak.notna())
    f["strong_wind_sum6h"] = roll(strong, 6, "sum")

    pos = pd.Series(np.arange(len(h), dtype=float), index=h.index)
    last_ts = pos.where(f["ts"] == 1).ffill()
    f["hours_since_ts"] = (pos - last_ts).clip(upper=HOURS_SINCE_CAP).fillna(HOURS_SINCE_CAP)
    f["temp_max6h_minus_now"] = roll(f["temp_c"], 6, "max") - f["temp_c"]

    idx = h.index
    solar_hour = (idx.hour + idx.minute / 60.0 + longitude / 15.0) % 24.0
    f["solar_hour_sin"] = np.sin(2 * np.pi * solar_hour / 24.0)
    f["solar_hour_cos"] = np.cos(2 * np.pi * solar_hour / 24.0)
    f["doy_sin"] = np.sin(2 * np.pi * idx.day_of_year / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.day_of_year / 365.25)
    f["latitude"] = float(latitude)
    f["longitude"] = float(longitude)
    f["elevation_m"] = float(elevation_m or 0.0)
    f["n_reports"] = h["n_reports"].astype(float)
    return f[FEATURE_COLUMNS]


def _future_any(event: pd.Series, reported: pd.Series, horizon: int) -> pd.Series:
    """1 if the event occurs in (h, h+H]; 0 if it does not and every hour of the
    window has a report; NaN (unknown) otherwise."""
    fut_max = event[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    n_rep = reported.astype(float)[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    label = pd.Series(np.nan, index=event.index)
    label[(fut_max == 0) & (n_rep == horizon)] = 0.0
    label[fut_max == 1] = 1.0
    return label


def build_labels(hourly: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking binary labels for every target in TARGETS."""
    reported = hourly["n_reports"] > 0
    peak = peak_wind(hourly)
    events = {
        "thunder": hourly["ts"].astype(float),
        "wind": (peak >= STRONG_WIND_MS).astype(float).where(peak.notna()),
    }
    out = pd.DataFrame(index=hourly.index)
    for name, (kind, horizon) in TARGETS.items():
        out[name] = _future_any(events[kind], reported, horizon)
    return out
