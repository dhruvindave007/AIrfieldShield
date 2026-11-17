import numpy as np
import pandas as pd
import pytest

from ai_models.ml.features import TARGETS
from ai_models.ml.metrics import (block_bootstrap, ci95, expected_calibration_error,
                                  precision_recall_at, summarize, threshold_for_precision)
from ai_models.ml.train import CellRate, time_split


def test_threshold_for_precision_picks_max_recall_point():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.linspace(0.0, 0.5, 80), np.linspace(0.3, 1.0, 20)])
    thr = threshold_for_precision(y, p, 0.5, min_alerts=5)
    prec, rec, n = precision_recall_at(y, p, thr)
    assert prec >= 0.5
    # any lower threshold that still alerts must violate the precision target
    lower = np.sort(np.unique(p[p < thr]))[-1]
    assert precision_recall_at(y, p, lower)[0] < 0.5


def test_threshold_unattainable_returns_none():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.02).astype(int)
    p = rng.random(2000)                       # no skill
    assert threshold_for_precision(y, p, 0.75) is None
    assert precision_recall_at(y, p, None) == (pytest.approx(float("nan"), nan_ok=True), 0.0, 0)


def test_summarize_perfect_and_climatological_forecasts():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    clim = np.full(10, 0.2)
    perfect = summarize(y, y.astype(float), clim)
    assert perfect["pr_auc"] == 1.0 and perfect["brier"] == 0.0 and perfect["bss"] == 1.0
    base = summarize(y, clim, clim)
    assert base["bss"] == pytest.approx(0.0)
    assert base["prevalence"] == 0.2


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    assert expected_calibration_error(y, np.full(5000, 0.1)) < 0.02
    assert expected_calibration_error(y, np.full(5000, 0.6)) > 0.45


def test_block_bootstrap_is_seeded_and_paired():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.2).astype(int)
    good = np.clip(y * 0.6 + rng.random(600) * 0.4, 0, 1)
    bad = rng.random(600)
    blocks = np.repeat(np.arange(60), 10)
    metric = lambda yy, pp: float(np.mean((pp - yy) ** 2))  # noqa: E731
    a = block_bootstrap(y, {"good": good, "bad": bad}, blocks, metric, n_boot=50, seed=1)
    b = block_bootstrap(y, {"good": good, "bad": bad}, blocks, metric, n_boot=50, seed=1)
    np.testing.assert_array_equal(a["good"], b["good"])
    lo, hi = ci95(a["good"] - a["bad"])
    assert hi < 0                                # good forecast has the lower Brier score


def _examples():
    t = pd.date_range("2021-12-30", "2023-01-03", freq="h", tz="UTC")
    return pd.DataFrame({"time": t, "station": "VECC", "y": 0.0})


@pytest.mark.parametrize("target", list(TARGETS))
def test_time_split_is_chronological_and_purged(target):
    horizon = TARGETS[target][1]
    parts = time_split(_examples(), horizon)
    gap = pd.Timedelta(hours=horizon)
    assert parts["train"]["time"].max() < parts["val"]["time"].min()
    assert parts["val"]["time"].max() < parts["test"]["time"].min()
    # the label window (h, h+H] of the last train/val row ends before the next split starts
    assert parts["train"]["time"].max() + gap < pd.Timestamp("2022-01-01", tz="UTC")
    assert parts["val"]["time"].max() + gap < pd.Timestamp("2023-01-01", tz="UTC")
    assert parts["test"]["time"].min() == pd.Timestamp("2023-01-01", tz="UTC")


def test_cell_rate_baseline_shrinks_towards_global_rate():
    df = pd.DataFrame({"k": ["a"] * 100 + ["b"] * 2})
    y = np.array([0] * 90 + [1] * 10 + [1, 1])
    m = CellRate(lambda d: d[["k"]], k=20.0).fit(df, y)
    pa, pb, punseen = m.predict(pd.DataFrame({"k": ["a", "b", "zzz"]}))
    global_rate = y.mean()
    assert pa == pytest.approx((10 + 20 * global_rate) / 120)
    assert global_rate < pb < 1.0                # two positives do not give probability 1
    assert punseen == pytest.approx(global_rate)
