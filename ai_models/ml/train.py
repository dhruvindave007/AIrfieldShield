#!/usr/bin/env python3
"""Reproducible training + evaluation of the AirfieldShield nowcasting models.

    python -m ai_models.ml.train                      # full run (data/metar_hourly.csv.gz)
    python -m ai_models.ml.train --quick              # smaller/faster smoke run

Protocol
--------
* chronological split:  train < 2022-01-01 <= validation < 2023-01-01 <= test
* rows whose label window crosses a split boundary are purged
* validation year is used for: boosting-iteration selection, class-weight
  selection, isotonic calibration and alert-threshold selection
* the test years are touched exactly once, for the numbers in the report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, STRONG_WIND_KT, STRONG_WIND_MS, TARGETS, build_features, build_labels, peak_wind
from .metrics import (PRECISION_TARGETS, block_bootstrap, ci95, precision_recall_at,
                      reliability_table, summarize, threshold_for_precision)
from .model import ARTIFACT_DIR, ARTIFACT_NAME, CalibratedNowcaster, save_bundle
from .report import write_figures, write_report
from .stations import STATIONS

ROOT = Path(__file__).resolve().parents[2]
TRAIN_END = "2022-01-01"
VAL_END = "2023-01-01"
GALE_MS = 34 * 0.514444


# --------------------------------------------------------------------------- data
def load_examples(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (examples, data_summary). One example per station-hour with a report."""
    hourly_all = pd.read_csv(data_path, parse_dates=["time"])
    frames, summary = [], []
    for station, h in hourly_all.groupby("station", sort=True):
        name, lat, lon, elev = STATIONS[station]
        h = h.set_index("time").drop(columns="station").sort_index()
        feats = build_features(h, lat, lon, elev)
        labels = build_labels(h)
        df = pd.concat([feats, labels], axis=1)
        df.insert(0, "station", station)
        df = df[h["n_reports"] > 0]
        frames.append(df.reset_index())
        pk = peak_wind(h)
        summary.append({
            "station": station, "name": name, "hours": len(h),
            "coverage": float((h["n_reports"] > 0).mean()),
            "ts_hours": int((h["ts"] == 1).sum()),
            "strong_wind_hours": int((pk >= STRONG_WIND_MS).sum()),
            "gale_hours": int((pk >= GALE_MS).sum()),
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary)


def time_split(df: pd.DataFrame, horizon_h: int) -> dict[str, pd.DataFrame]:
    """Chronological split; purge rows whose label window (h, h+H] crosses a boundary."""
    t = df["time"]
    train_end, val_end = pd.Timestamp(TRAIN_END, tz="UTC"), pd.Timestamp(VAL_END, tz="UTC")
    gap = pd.Timedelta(hours=horizon_h)
    return {
        "train": df[t < train_end - gap],
        "val": df[(t >= train_end) & (t < val_end - gap)],
        "test": df[t >= val_end],
    }


# ---------------------------------------------------------------------- baselines
class CellRate:
    """Empirical event rate per cell of discrete keys, shrunk toward the global
    rate with ``k`` pseudo-counts. Used for climatology and persistence baselines."""

    def __init__(self, key_fn, k: float = 20.0):
        self.key_fn, self.k = key_fn, k

    def fit(self, df, y):
        keys = self.key_fn(df)
        self.cols_ = list(keys.columns)
        self.global_ = float(np.mean(y))
        g = keys.assign(_y=np.asarray(y)).groupby(self.cols_)["_y"].agg(["sum", "count"])
        self.table_ = ((g["sum"] + self.k * self.global_) / (g["count"] + self.k)).rename("_p")
        return self

    def predict(self, df):
        keys = self.key_fn(df)
        p = keys.join(self.table_, on=self.cols_)["_p"]
        return p.fillna(self.global_).to_numpy()


def _solar_bin(df):
    hour = (np.degrees(np.arctan2(df["solar_hour_sin"], df["solar_hour_cos"])) / 15.0) % 24.0
    return (np.round(hour).astype(int) % 24) // 3


def climatology_keys(df):
    return pd.DataFrame({"station": df["station"].to_numpy(), "month": df["time"].dt.month.to_numpy(),
                         "hbin": _solar_bin(df).to_numpy()}, index=df.index)


def persistence_keys(kind):
    def fn(df):
        if kind == "thunder":
            now = df["ts"].fillna(0) > 0
            recent = df["ts_sum3h"].fillna(0) > 0
        else:
            now = df["peak_wind_ms"].fillna(0) >= STRONG_WIND_MS
            recent = df["peak_wind_max3h"].fillna(0) >= STRONG_WIND_MS
        return pd.DataFrame({"now": now.astype(int), "recent": recent.astype(int)}, index=df.index)
    return fn


def trend_keys(df):
    return pd.DataFrame({"now": (df["ts"].fillna(0) > 0).astype(int),
                         "trend": (df["trend_ts"].fillna(0) > 0).astype(int)}, index=df.index)


def onset_mask(df, kind):
    """Rows where the hazard is neither occurring now nor in the previous 3 h."""
    if kind == "thunder":
        return (df["ts_sum3h"].fillna(0) == 0).to_numpy()
    return (df["peak_wind_max3h"].fillna(0) < STRONG_WIND_MS).to_numpy()


# ------------------------------------------------------------------------- models
def fit_hgb(Xtr, ytr, Xva, yva, features, seed, max_iter, class_weight):
    """Fit HGB on train; pick the boosting iteration maximising validation PR-AUC
    (time-ordered early stopping instead of sklearn's random internal split)."""
    params = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, class_weight=class_weight,
                  early_stopping=False, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=max_iter, **params).fit(Xtr[features], ytr)
    best_ap, best_iter = -1.0, max_iter
    for i, proba in enumerate(clf.staged_predict_proba(Xva[features]), start=1):
        if i % 10 == 0 or i == max_iter:
            ap = average_precision_score(yva, proba[:, 1])
            if ap > best_ap + 1e-6:
                best_ap, best_iter = ap, i
    if best_iter != max_iter:
        clf = HistGradientBoostingClassifier(max_iter=best_iter, **params).fit(Xtr[features], ytr)
    return clf, best_iter, float(best_ap)


def fit_isotonic(p_val, y_val):
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)


def run_target(name, df, seed, max_iter, n_boot, log):
    kind, horizon = TARGETS[name]
    data = df[df[name].notna()]
    parts = time_split(data, horizon)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    ytr, yva, yte = (p[name].astype(int).to_numpy() for p in (tr, va, te))
    log(f"[{name}] train={len(tr)} ({ytr.mean():.4f})  val={len(va)} ({yva.mean():.4f})  "
        f"test={len(te)} ({yte.mean():.4f})")

    preds_val, preds_test = {}, {}

    # --- baselines (fitted on train only)
    base = {
        "Base rate (constant)": CellRate(lambda d: pd.DataFrame({"c": 0}, index=d.index)),
        "Climatology (station x month x 3h)": CellRate(climatology_keys),
        "Persistence": CellRate(persistence_keys(kind), k=1.0),
    }
    if kind == "thunder":
        base["Persistence + METAR trend"] = CellRate(trend_keys, k=1.0)
    for label, m in base.items():
        m.fit(tr, ytr)
        preds_val[label], preds_test[label] = m.predict(va), m.predict(te)

    # --- logistic regression (linear reference model)
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                          random_state=seed))
    lr.fit(tr[FEATURE_COLUMNS], ytr)
    lr_cal = fit_isotonic(lr.predict_proba(va[FEATURE_COLUMNS])[:, 1], yva)
    preds_val["Logistic regression"] = lr_cal.predict(lr.predict_proba(va[FEATURE_COLUMNS])[:, 1])
    preds_test["Logistic regression"] = lr_cal.predict(lr.predict_proba(te[FEATURE_COLUMNS])[:, 1])

    # --- gradient boosting: class-imbalance handling chosen on validation
    candidates = {}
    for cw in (None, "balanced"):
        clf, it, ap = fit_hgb(tr, ytr, va, yva, FEATURE_COLUMNS, seed, max_iter, cw)
        candidates[str(cw)] = {"clf": clf, "best_iter": it, "val_pr_auc": ap}
        log(f"[{name}] HGB class_weight={cw}: best_iter={it} val PR-AUC={ap:.4f}")
    chosen_cw = max(candidates, key=lambda k: candidates[k]["val_pr_auc"])
    clf = candidates[chosen_cw]["clf"]
    raw_val = clf.predict_proba(va[FEATURE_COLUMNS])[:, 1]
    raw_test = clf.predict_proba(te[FEATURE_COLUMNS])[:, 1]
    iso = fit_isotonic(raw_val, yva)
    HGB = "HistGradientBoosting (calibrated)"
    preds_val[HGB], preds_test[HGB] = iso.predict(raw_val), iso.predict(raw_test)
    preds_test["HistGradientBoosting (uncalibrated)"] = raw_test

    # --- ablation: drop the human forecaster's trend flag
    ablation = None
    if kind == "thunder":
        feats_nt = [f for f in FEATURE_COLUMNS if f != "trend_ts"]
        cw = None if chosen_cw == "None" else chosen_cw
        clf_nt, _, _ = fit_hgb(tr, ytr, va, yva, feats_nt, seed, max_iter, cw)
        iso_nt = fit_isotonic(clf_nt.predict_proba(va[feats_nt])[:, 1], yva)
        p_nt = iso_nt.predict(clf_nt.predict_proba(te[feats_nt])[:, 1])
        preds_test["HGB without trend_ts (ablation)"] = p_nt

    # --- headline metrics on test
    clim = preds_test["Climatology (station x month x 3h)"]
    metrics = {k: summarize(yte, p, clim) for k, p in preds_test.items()}

    # --- uncertainty: day-block bootstrap, paired across methods
    days = (te["station"] + te["time"].dt.strftime("%Y%m%d")).to_numpy()
    boot_ap = block_bootstrap(yte, preds_test, days, average_precision_score, n_boot, seed)
    boot_br = block_bootstrap(yte, preds_test, days, brier_score_loss, n_boot, seed)
    for k in preds_test:
        metrics[k]["pr_auc_ci"] = ci95(boot_ap[k])
        metrics[k]["brier_ci"] = ci95(boot_br[k])
    paired = {k: {"pr_auc_diff_ci": ci95(boot_ap[HGB] - boot_ap[k]),
                  "brier_diff_ci": ci95(boot_br[HGB] - boot_br[k])}
              for k in preds_test if k != HGB}

    # --- operating points: thresholds from validation, performance on test
    thresholds, operating = {}, []
    for target in PRECISION_TARGETS:
        key = f"p{int(target * 100)}"
        thr = threshold_for_precision(yva, preds_val[HGB], target)
        thresholds[key] = thr
        vp, vr, _ = precision_recall_at(yva, preds_val[HGB], thr)
        tp_, tr_, tn = precision_recall_at(yte, preds_test[HGB], thr)
        operating.append({"precision_target": target, "threshold": thr, "val_precision": vp,
                          "val_recall": vr, "test_precision": tp_, "test_recall": tr_,
                          "test_alerts": tn})

    # --- onset-only subset (hazard not present now or in the last 3 h)
    om = onset_mask(te, kind)
    onset = {"n": int(om.sum()), "positives": int(yte[om].sum()),
             "prevalence": float(yte[om].mean()),
             "pr_auc": {k: float(average_precision_score(yte[om], np.asarray(p)[om]))
                        for k, p in preds_test.items()}}

    # --- per-station (calibrated HGB)
    per_station = []
    for st, idx in te.groupby("station").indices.items():
        if yte[idx].sum() >= 5:
            row = summarize(yte[idx], preds_test[HGB][idx], clim[idx])
            row["station"] = st
            per_station.append(row)

    # --- permutation importance on (a subsample of) the test period
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(te), size=min(len(te), 40000), replace=False)
    pi = permutation_importance(clf, te[FEATURE_COLUMNS].iloc[sub], yte[sub],
                                scoring="average_precision", n_repeats=3, random_state=seed)
    order = np.argsort(-pi.importances_mean)[:10]
    importance = [(FEATURE_COLUMNS[i], float(pi.importances_mean[i]), float(pi.importances_std[i]))
                  for i in order]

    result = {
        "target": name, "kind": kind, "horizon_h": horizon,
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "positives": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())},
        "metrics": metrics, "paired_vs_hgb": paired, "operating_points": operating,
        "thresholds": thresholds, "onset": onset, "per_station": per_station,
        "importance": importance,
        "hgb_selection": {k: {"best_iter": v["best_iter"], "val_pr_auc": v["val_pr_auc"]}
                          for k, v in candidates.items()},
        "chosen_class_weight": chosen_cw,
        "reliability": {"calibrated": reliability_table(yte, preds_test[HGB]),
                        "uncalibrated": reliability_table(yte, raw_test)},
    }
    curves = {"y": yte, "preds": preds_test}
    model = CalibratedNowcaster(clf, iso, FEATURE_COLUMNS, thresholds)
    return model, result, curves


# --------------------------------------------------------------------------- main
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_qc(data_path: Path) -> dict:
    meta = data_path.with_name("metar_hourly.meta.json")
    return json.loads(meta.read_text(encoding="utf-8")).get("qc", {}) if meta.exists() else {}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "metar_hourly.csv.gz")
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "evaluation_report.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="fast smoke run (fewer iterations/bootstraps)")
    args = ap.parse_args(argv)
    if args.quick:
        args.max_iter, args.n_boot = 60, 20
    np.random.seed(args.seed)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    df, data_summary = load_examples(args.data)
    log(f"loaded {len(df)} station-hours from {args.data.name}")

    models, results, curves = {}, {}, {}
    for name in TARGETS:
        models[name], results[name], curves[name] = run_target(
            name, df, args.seed, args.max_iter, args.n_boot, log)

    metadata = {
        "artifact": ARTIFACT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "quick_mode": bool(args.quick),
        "data": {"file": args.data.name, "sha256": _sha256(args.data),
                 "source": "Iowa Environmental Mesonet ASOS/METAR archive",
                 "stations": list(STATIONS), "first": str(df["time"].min()),
                 "last": str(df["time"].max()), "station_hours": int(len(df))},
        "data_qc": _load_qc(args.data),
        "split": {"train": f"< {TRAIN_END}", "validation": f"{TRAIN_END} .. {VAL_END}",
                  "test": f">= {VAL_END}", "purge": "label window H at each boundary"},
        "strong_wind_threshold_kt": STRONG_WIND_KT,
        "feature_columns": FEATURE_COLUMNS,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "targets": results,
    }
    bundle = {"models": models, "feature_columns": FEATURE_COLUMNS}
    path = save_bundle(bundle, metadata, args.artifact_dir)
    log(f"saved {path} ({path.stat().st_size / 1e3:.0f} kB)")

    fig_dir = args.report.parent / "figures"
    write_figures(curves, results, fig_dir)
    write_report(args.report, metadata, data_summary)
    log(f"wrote {args.report}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
