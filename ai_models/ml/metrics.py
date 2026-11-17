"""Evaluation metrics for rare-event probabilistic forecasts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

PRECISION_TARGETS = (0.25, 0.50, 0.75)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-mass bins (equal-width bins are useless for rare events,
    where almost all forecasts fall into the lowest bin)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def reliability_table(y, p, edges=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0000001)):
    """Rows of (bin_low, bin_high, n, mean_forecast, observed_frequency)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append((lo, min(hi, 1.0), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def threshold_for_precision(y, p, target: float, min_alerts: int = 20):
    """Lowest threshold whose precision on (y, p) is >= target (i.e. the
    max-recall operating point meeting the precision constraint). Returns None
    if unattainable with at least ``min_alerts`` alerts."""
    precision, recall, thr = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    p = np.asarray(p)
    ok = precision >= target
    best = None
    for i in np.where(ok)[0]:
        if (p >= thr[i]).sum() >= min_alerts:
            best = i
            break  # thresholds ascend, so the first admissible one has max recall
    return None if best is None else float(thr[best])


def precision_recall_at(y, p, threshold):
    """(precision, recall, n_alerts) when alerting on p >= threshold."""
    if threshold is None:
        return (float("nan"), 0.0, 0)
    y = np.asarray(y).astype(bool)
    alert = np.asarray(p) >= threshold
    n = int(alert.sum())
    tp = int((alert & y).sum())
    return (tp / n if n else float("nan"), tp / max(int(y.sum()), 1), n)


def summarize(y, p, p_reference=None) -> dict:
    """Headline metrics. ``p_reference`` (e.g. climatology) enables Brier skill."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
    if p_reference is not None:
        ref = brier_score_loss(y, np.clip(p_reference, 0, 1))
        out["bss"] = float(1.0 - out["brier"] / ref) if ref > 0 else float("nan")
    return out


def block_bootstrap(y, preds: dict, blocks, metric, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap replicates of ``metric`` for each forecast in ``preds``, resampling
    whole blocks (calendar days) with replacement because hourly weather samples
    are strongly autocorrelated. The same resamples are used for every forecast,
    so differences between the returned arrays are paired."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    codes, uniq = _factorize(blocks)
    order = np.argsort(codes, kind="mergesort")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    stats = {k: [] for k in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[b]:bounds[b + 1]] for b in pick])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        for k, p in preds.items():
            stats[k].append(metric(yb, np.asarray(p)[idx]))
    return {k: np.asarray(v) for k, v in stats.items()}


def ci95(samples) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _factorize(values):
    uniq, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes, uniq
