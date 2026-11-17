"""Turn irregular METAR reports into a regular hourly table per station.

Canonical observation columns (one row per report):
    valid (UTC), temp_c, dewpoint_c, rh, wind_dir, wind_ms, gust_ms,
    pressure_hpa, visibility_km, ts, cb, precip, trend_ts

Hourly bucket ``h`` aggregates all reports with  h-1h < valid <= h, so a row
stamped ``h`` only ever contains information available at time ``h``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .metar import parse_metar_flags, split_metar

KT_TO_MS = 0.514444
INHG_TO_HPA = 33.8639
MILE_TO_KM = 1.609344

FLAG_COLS = ["ts", "cb", "precip", "trend_ts"]
OBS_COLS = ["temp_c", "dewpoint_c", "rh", "wind_dir", "wind_ms", "gust_ms",
            "pressure_hpa", "visibility_km"] + FLAG_COLS
HOURLY_COLS = OBS_COLS + ["n_reports"]

_FLAG_SOURCE = {"ts": "ts_obs", "cb": "cb_obs", "precip": "precip_obs", "trend_ts": "trend_ts"}


def raw_iem_to_obs(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert an IEM ASOS-archive CSV frame to canonical SI observations."""
    flags = pd.DataFrame([parse_metar_flags(m) for m in raw["metar"]], index=raw.index)
    obs = pd.DataFrame({
        "valid": pd.to_datetime(raw["valid"], utc=True),
        "temp_c": (raw["tmpf"] - 32.0) / 1.8,
        "dewpoint_c": (raw["dwpf"] - 32.0) / 1.8,
        "rh": raw["relh"],
        "wind_dir": raw["drct"],
        "wind_ms": raw["sknt"] * KT_TO_MS,
        "gust_ms": raw["gust"] * KT_TO_MS,
        "pressure_hpa": raw["alti"] * INHG_TO_HPA,
        "visibility_km": raw["vsby"] * MILE_TO_KM,
    })
    for c in FLAG_COLS:
        obs[c] = flags[_FLAG_SOURCE[c]].astype(float)
    obs["metar"] = raw["metar"]
    obs = obs.dropna(subset=["valid"]).drop_duplicates("valid").sort_values("valid")
    return quality_control(obs)


_WIND_TOKEN = re.compile(r"^(?:\d{3}|VRB)\d{2}(?:G\d{2})?KT$")
MAX_WIND_MS = 60 * KT_TO_MS
MAX_GUST_MS = 80 * KT_TO_MS
SPIKE_MIN_MS = 25 * KT_TO_MS


def _wind_token_ok(metar) -> bool:
    """True if the report has a well-formed wind group. Hand-typed METARs contain
    groups such as ``070064KT`` or ``25008G180KT`` that decode to absurd speeds."""
    if not isinstance(metar, str):
        return True  # no raw text to check against (non-METAR source)
    body = split_metar(metar)[0]
    tokens = [t for t in body.split() if t.endswith("KT")]
    return bool(tokens) and bool(_WIND_TOKEN.match(tokens[0]))


def quality_control(obs: pd.DataFrame) -> pd.DataFrame:
    """Causal quality control (uses the report itself and the previous report only,
    so the identical procedure runs at serving time). Suspect values become NaN;
    counts are stored in ``obs.attrs['qc']``."""
    obs = obs.copy()
    qc = {}
    wind_cols = ["wind_ms", "gust_ms", "wind_dir"]

    bad_token = ~obs["metar"].map(_wind_token_ok) if "metar" in obs else pd.Series(False, index=obs.index)
    bad_token &= obs["wind_ms"].notna()
    qc["malformed_wind_group"] = int(bad_token.sum())
    obs.loc[bad_token, wind_cols] = np.nan

    bad_range = (obs["wind_ms"] > MAX_WIND_MS) | (obs["wind_ms"] < 0)
    qc["wind_out_of_range"] = int(bad_range.sum())
    obs.loc[bad_range, wind_cols] = np.nan
    bad_gust = (obs["gust_ms"] > MAX_GUST_MS) | (obs["gust_ms"] < obs["wind_ms"])
    qc["gust_out_of_range"] = int(bad_gust.sum())
    obs.loc[bad_gust, "gust_ms"] = np.nan

    # Implausible gust factor, e.g. "09006G46KT ... BR" (fog, 6 kt, "gust 46").
    convective = (obs["ts"] == 1) | (obs["cb"] == 1)
    spread = obs["gust_ms"] - obs["wind_ms"]
    bad_factor = (obs["gust_ms"] >= SPIKE_MIN_MS) & (
        ((spread >= 30 * KT_TO_MS) & ~convective)
        | ((obs["gust_ms"] > 3.0 * obs["wind_ms"].clip(lower=5 * KT_TO_MS))
           & ~convective & (obs["precip"] != 1)))
    qc["implausible_gust_factor"] = int(bad_factor.sum())
    obs.loc[bad_factor, "gust_ms"] = np.nan

    # Isolated strong sustained wind: >= 25 kt with no gust group, no convective
    # or precipitation evidence in the same report, and more than 2.5x the
    # previous report (<= 90 min earlier). Typical cause: "04040KT" for "04004KT".
    prev_wind = obs["wind_ms"].shift(1)
    prev_gap = obs["valid"].diff() <= pd.Timedelta(minutes=90)
    evidence = (obs["ts"] == 1) | (obs["cb"] == 1) | (obs["precip"] == 1) | obs["gust_ms"].notna()
    spike = ((obs["wind_ms"] >= SPIKE_MIN_MS) & ~evidence & prev_gap
             & (prev_wind.notna()) & (obs["wind_ms"] > 2.5 * prev_wind.clip(lower=2.0)))
    qc["isolated_wind_spike"] = int(spike.sum())
    obs.loc[spike, wind_cols] = np.nan

    for col, lo, hi in (("temp_c", -40, 55), ("dewpoint_c", -60, 40), ("pressure_hpa", 900, 1060)):
        bad = obs[col].notna() & ~obs[col].between(lo, hi)
        qc[f"{col}_out_of_range"] = int(bad.sum())
        obs.loc[bad, col] = np.nan
    jump = prev_gap & ((obs["temp_c"] - obs["temp_c"].shift(1)).abs() > 15)
    qc["temp_jump"] = int(jump.sum())
    obs.loc[jump, ["temp_c", "dewpoint_c", "rh"]] = np.nan
    bad_dp = obs["dewpoint_c"] > obs["temp_c"] + 1.0
    qc["dewpoint_above_temp"] = int(bad_dp.sum())
    obs.loc[bad_dp, ["dewpoint_c", "rh"]] = np.nan

    qc["reports"] = int(len(obs))
    obs.attrs["qc"] = qc
    return obs


def build_hourly(obs: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Aggregate canonical observations to a gap-free hourly grid (UTC index 'time')."""
    obs = obs.copy()
    for c in OBS_COLS:
        if c not in obs:
            obs[c] = np.nan
    obs[OBS_COLS] = obs[OBS_COLS].astype(float)
    obs["valid"] = pd.to_datetime(obs["valid"], utc=True)
    obs["time"] = obs["valid"].dt.ceil("h")
    obs = obs.sort_values("valid")
    agg = {
        "temp_c": "last", "dewpoint_c": "last", "rh": "last", "wind_dir": "last",
        "pressure_hpa": "last", "wind_ms": "max", "gust_ms": "max",
        "visibility_km": "min", "ts": "max", "cb": "max", "precip": "max",
        "trend_ts": "max",
    }
    g = obs.groupby("time")
    hourly = g.agg(agg)
    hourly["n_reports"] = g.size()
    if len(hourly) == 0 and (start is None or end is None):
        empty = pd.DataFrame(columns=HOURLY_COLS,
                             index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float)
        return empty
    start = hourly.index.min() if start is None else _utc(start).ceil("h")
    end = hourly.index.max() if end is None else _utc(end).ceil("h")
    grid = pd.date_range(start, end, freq="h", name="time")
    hourly = hourly.reindex(grid)
    hourly["n_reports"] = hourly["n_reports"].fillna(0)
    return hourly[HOURLY_COLS].astype(float)


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
