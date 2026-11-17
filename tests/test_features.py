"""Feature engineering: correctness and, above all, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from ai_models.ml.dataset import build_hourly, quality_control, raw_iem_to_obs
from ai_models.ml.features import FEATURE_COLUMNS, STRONG_WIND_MS, build_features, build_labels
from ai_models.ml.metar import parse_metar_flags, split_metar


def make_hourly(n=96, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC", name="time")
    h = pd.DataFrame({
        "temp_c": 30 + rng.normal(0, 2, n),
        "dewpoint_c": 22 + rng.normal(0, 1, n),
        "rh": 60 + rng.normal(0, 5, n),
        "wind_dir": rng.uniform(0, 360, n),
        "wind_ms": rng.uniform(0, 8, n),
        "gust_ms": np.nan,
        "pressure_hpa": 1005 + rng.normal(0, 1, n),
        "visibility_km": 6.0,
        "ts": (rng.random(n) < 0.1).astype(float),
        "cb": (rng.random(n) < 0.2).astype(float),
        "precip": 0.0,
        "trend_ts": 0.0,
        "n_reports": 2.0,
    }, index=idx)
    return h


# ---------------------------------------------------------------- METAR parsing
def test_metar_flags_observed_thunderstorm_and_cb():
    f = parse_metar_flags("VIDP 152130Z 32030KT 3200 TSRA SCT035 FEW040CB BKN100 29/19 Q1004 NOSIG")
    assert f == {"ts_obs": True, "cb_obs": True, "precip_obs": True, "trend_ts": False}


def test_metar_trend_thunderstorm_is_not_an_observation():
    raw = "VIDP 010000Z 30004KT 3000 HZ SCT030 FEW035CB BKN100 34/11 Q1000 TEMPO 29015G25KT 1500 TS DS"
    f = parse_metar_flags(raw)
    assert f["ts_obs"] is False          # the TS is only forecast in the trend
    assert f["trend_ts"] is True
    assert split_metar(raw)[0].endswith("Q1000")


def test_metar_vicinity_thunderstorm_and_plain_report():
    assert parse_metar_flags("VABB 010000Z 25008KT 4000 VCTS FEW030CB 29/25 Q1003")["ts_obs"]
    plain = parse_metar_flags("VOBL 240900Z 05005KT 8000 NSC 29/15 Q1013 NOSIG")
    assert not any(plain.values())
    assert parse_metar_flags(None) == {"ts_obs": False, "cb_obs": False,
                                       "precip_obs": False, "trend_ts": False}


# ------------------------------------------------------------- quality control
def _raw(rows):
    return pd.DataFrame(rows, columns=["valid", "tmpf", "dwpf", "relh", "drct", "sknt", "gust",
                                       "alti", "vsby", "metar"])


def test_qc_rejects_malformed_and_spiking_wind_groups():
    raw = _raw([
        ["2024-06-01 00:00", 86, 70, 60, 70, 4, np.nan, 29.6, 3, "VECC 010000Z 07004KT 5000 HZ NSC 30/21 Q1002"],
        # six-digit wind group decoded by the archive as 64 kt
        ["2024-06-01 00:30", 86, 70, 60, 70, 64, np.nan, 29.6, 3, "VECC 010030Z 070064KT 5000 HZ NSC 30/21 Q1002"],
        ["2024-06-01 01:00", 86, 70, 60, 70, 4, np.nan, 29.6, 3, "VECC 010100Z 07004KT 5000 HZ NSC 30/21 Q1002"],
        # well-formed but isolated 40 kt in haze after 4 kt: typo for 04004KT
        ["2024-06-01 01:30", 86, 70, 60, 40, 40, np.nan, 29.6, 3, "VECC 010130Z 04040KT 5000 HZ NSC 30/21 Q1002"],
        # genuine squall: gust group and thunderstorm in the same report
        ["2024-06-01 02:00", 77, 70, 80, 300, 30, 45, 29.6, 1, "VECC 010200Z 30030G45KT 0900 +TSRA BKN015 FEW025CB 25/21 Q1004"],
    ])
    obs = raw_iem_to_obs(raw)
    assert np.isnan(obs["wind_ms"].iloc[1])
    assert np.isnan(obs["wind_ms"].iloc[3])
    assert obs["wind_ms"].iloc[4] == pytest.approx(30 * 0.514444)
    assert obs["gust_ms"].iloc[4] == pytest.approx(45 * 0.514444)
    assert obs.attrs["qc"]["malformed_wind_group"] == 1
    assert obs.attrs["qc"]["isolated_wind_spike"] == 1


def test_qc_is_causal():
    """Appending later reports must never change the QC outcome of earlier ones."""
    rng = np.random.default_rng(1)
    n = 60
    obs = pd.DataFrame({
        "valid": pd.date_range("2024-06-01", periods=n, freq="30min", tz="UTC"),
        "temp_c": 30 + rng.normal(0, 8, n), "dewpoint_c": 20.0, "rh": 50.0,
        "wind_dir": 90.0, "wind_ms": rng.choice([2.0, 3.0, 14.0, 20.0], n), "gust_ms": np.nan,
        "pressure_hpa": 1005.0, "visibility_km": 5.0,
        "ts": 0.0, "cb": 0.0, "precip": 0.0, "trend_ts": 0.0,
    })
    full = quality_control(obs)
    part = quality_control(obs.iloc[:40])
    pd.testing.assert_frame_equal(full.iloc[:40].reset_index(drop=True),
                                  part.reset_index(drop=True))


# ------------------------------------------------------------ hourly aggregation
def test_hourly_bucket_contains_only_past_reports():
    obs = pd.DataFrame({
        "valid": pd.to_datetime(["2024-06-01 00:30", "2024-06-01 01:00", "2024-06-01 01:01",
                                 "2024-06-01 03:00"], utc=True),
        "temp_c": [30.0, 31.0, 32.0, 33.0], "wind_ms": [1.0, 5.0, 9.0, 2.0],
        "ts": [0.0, 0.0, 1.0, 0.0],
    })
    h = build_hourly(obs)
    assert list(h.index.strftime("%H:%M")) == ["01:00", "02:00", "03:00"]
    assert h.loc["2024-06-01 01:00", "temp_c"] == 31.0      # last report at or before 01:00
    assert h.loc["2024-06-01 01:00", "wind_ms"] == 5.0      # max within (00:00, 01:00]
    assert h.loc["2024-06-01 01:00", "ts"] == 0.0           # the 01:01 TS belongs to 02:00
    assert h.loc["2024-06-01 02:00", "ts"] == 1.0
    assert h.loc["2024-06-01 02:00", "n_reports"] == 1
    assert h["n_reports"].tolist() == [2, 1, 1]


def test_hourly_fills_gaps_with_missing_rows():
    obs = pd.DataFrame({"valid": pd.to_datetime(["2024-06-01 01:00", "2024-06-01 05:00"], utc=True),
                        "temp_c": [30.0, 25.0]})
    h = build_hourly(obs)
    assert len(h) == 5
    assert h["n_reports"].tolist() == [1, 0, 0, 0, 1]
    assert h["temp_c"].isna().sum() == 3


# -------------------------------------------------------------------- features
def test_feature_columns_and_values():
    h = make_hourly()
    f = build_features(h, latitude=22.65, longitude=88.45, elevation_m=5)
    assert list(f.columns) == FEATURE_COLUMNS
    assert len(f) == len(h)
    t = h.index[10]
    assert f.loc[t, "pressure_d3h"] == pytest.approx(h["pressure_hpa"].iloc[10] - h["pressure_hpa"].iloc[7])
    assert f.loc[t, "ts_sum6h"] == h["ts"].iloc[5:11].sum()
    assert f.loc[t, "peak_wind_max3h"] == pytest.approx(h["wind_ms"].iloc[8:11].max())
    assert f.loc[t, "dewpoint_depression"] == pytest.approx(h["temp_c"].iloc[10] - h["dewpoint_c"].iloc[10])
    assert f["latitude"].eq(22.65).all()


def test_features_have_no_lookahead():
    """Features at hour h must be identical whether or not later data exists,
    and must not change when the future is replaced by arbitrary values."""
    h = make_hourly(n=120)
    cut = 80
    full = build_features(h, 22.65, 88.45, 5)
    truncated = build_features(h.iloc[:cut], 22.65, 88.45, 5)
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)

    tampered = h.copy()
    tampered.iloc[cut:, :] = 999.0
    pd.testing.assert_frame_equal(build_features(tampered, 22.65, 88.45, 5).iloc[:cut], truncated)


def test_hours_since_thunderstorm():
    h = make_hourly(n=10)
    h["ts"] = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    f = build_features(h, 0, 0)
    assert f["hours_since_ts"].tolist() == [72, 72, 0, 1, 2, 3, 0, 1, 2, 3]


def test_solar_time_uses_longitude():
    h = make_hourly(n=24)
    f = build_features(h, 0.0, 90.0)   # 90E -> local solar time = UTC + 6 h
    noon_utc = h.index[6]              # 06:00 UTC is solar noon at 90E
    assert f.loc[noon_utc, "solar_hour_cos"] == pytest.approx(-1.0)


def test_missing_gust_falls_back_to_sustained_wind():
    h = make_hourly(n=5)
    h["wind_ms"] = [5.0, 5.0, 5.0, 5.0, 5.0]
    h["gust_ms"] = [np.nan, 15.0, np.nan, np.nan, np.nan]
    f = build_features(h, 0, 0)
    assert f["peak_wind_ms"].tolist() == [5.0, 15.0, 5.0, 5.0, 5.0]


# ---------------------------------------------------------------------- labels
def test_labels_look_strictly_forward():
    h = make_hourly(n=8)
    h["ts"] = [0, 0, 0, 1, 0, 0, 0, 0]
    y = build_labels(h)
    # 1 h label: positive only at the hour immediately before the event
    assert y["thunder_1h"].tolist()[:7] == [0, 0, 1, 0, 0, 0, 0]
    # 3 h label: positive for the three hours before the event, not at the event hour
    assert y["thunder_3h"].tolist()[:5] == [1, 1, 1, 0, 0]
    # the last rows have an incomplete future window -> unknown
    assert np.isnan(y["thunder_1h"].iloc[-1])
    assert y["thunder_3h"].iloc[-3:].isna().all()


def test_label_unknown_when_future_hour_unreported():
    h = make_hourly(n=6)
    h["ts"] = 0.0
    h.iloc[3, h.columns.get_loc("n_reports")] = 0
    h.iloc[3, h.columns.get_loc("ts")] = np.nan
    y = build_labels(h)
    assert np.isnan(y["thunder_1h"].iloc[2])      # next hour has no report
    assert y["thunder_1h"].iloc[1] == 0.0
    assert np.isnan(y["thunder_3h"].iloc[0])      # window 1..3 contains the gap
    h.iloc[4, h.columns.get_loc("ts")] = 1.0
    assert build_labels(h)["thunder_3h"].iloc[1] == 1.0   # a seen event is positive regardless


def test_wind_label_uses_peak_of_gust_and_sustained():
    h = make_hourly(n=5)
    h["wind_ms"] = 3.0
    h["gust_ms"] = [np.nan, np.nan, STRONG_WIND_MS + 1, np.nan, np.nan]
    y = build_labels(h)
    assert y["wind_1h"].tolist()[:4] == [0, 1, 0, 0]
