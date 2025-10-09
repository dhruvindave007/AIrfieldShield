#!/usr/bin/env python3
"""
Management command: predict
Produces Prediction rows for each Airfield using either:
 - the trained ensemble artifacts in ai_models/trained/ (RF + LSTM + CNN + meta) if available
 - or fallback heuristic when models are absent
Robust to a mix of artifact formats and missing components.
"""
from datetime import timezone
from django.core.management.base import BaseCommand
from core.models import Airfield, WeatherObservation, Prediction
import numpy as np
import os
import joblib
import logging
from pathlib import Path
from django.utils import timezone


logger = logging.getLogger(__name__)

TRAINED_DIR = Path("ai_models/trained")

# Try to import TensorFlow / Keras but tolerate absence
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    tf = None
    keras_load_model = None

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning("Failed to load joblib %s: %s", path, e)
        return None

def safe_load_keras(path):
    if keras_load_model is None:
        return None
    try:
        return keras_load_model(path)
    except Exception as e:
        logger.warning("Failed to load keras model %s: %s", path, e)
        return None

def compute_tabular_from_obs(obs):
    """
    Build a tabular feature vector from a list of WeatherObservation objects (most recent first).
    Returns numpy array of shape (feat_dim,)
    """
    temps = [o.temperature_c for o in obs if o.temperature_c is not None]
    hums = [o.humidity for o in obs if o.humidity is not None]
    press = [o.pressure_hpa for o in obs if o.pressure_hpa is not None]
    wind = [o.wind_speed_ms for o in obs if o.wind_speed_ms is not None]
    gust = [o.wind_gust_ms for o in obs if o.wind_gust_ms is not None]

    def safe_stats(arr):
        if not arr:
            return [np.nan, np.nan, np.nan]
        a = np.array(arr, dtype=np.float32)
        return [float(np.nanmean(a)), float(np.nanstd(a)), float(np.nanmax(a))]

    t_mean, t_std, t_max = safe_stats(temps)
    h_mean, h_std, h_max = safe_stats(hums)
    p_mean, p_std, p_max = safe_stats(press)
    w_mean, w_std, w_max = safe_stats(wind)
    g_mean, g_std, g_max = safe_stats(gust)

    feats = np.array([
        t_mean, t_std, t_max,
        h_mean, h_std, h_max,
        p_mean, p_std, p_max,
        w_mean, w_std, w_max,
        g_mean, g_std, g_max
    ], dtype=np.float32)
    # NaNs -> zeros (so models don't crash)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

def build_sequence_from_obs(obs, seq_len=30):
    """
    Build (seq_len, obs_dim) sequence for LSTM from observations (most recent first).
    If fewer than seq_len, pad with last value or zeros.
    obs expected ordered desc by timestamp.
    obs_dim := [temp, humidity, pressure, wind_speed, gust]
    """
    seq = []
    for o in obs:
        seq.append([
            float(o.temperature_c) if o.temperature_c is not None else 0.0,
            float(o.humidity) if o.humidity is not None else 0.0,
            float(o.pressure_hpa) if o.pressure_hpa is not None else 0.0,
            float(o.wind_speed_ms) if o.wind_speed_ms is not None else 0.0,
            float(o.wind_gust_ms) if o.wind_gust_ms is not None else 0.0,
        ])
        if len(seq) >= seq_len:
            break
    # pad to seq_len
    if len(seq) < seq_len:
        if seq:
            last = seq[-1]
        else:
            last = [0.0]*5
        while len(seq) < seq_len:
            seq.append(last)
    return np.array(seq, dtype=np.float32)[::-1]  # reverse so oldest->newest

class Command(BaseCommand):
    help = "Run lightweight/ensemble predictions for all airfields"

    def add_arguments(self, parser):
        parser.add_argument("--seq-len", type=int, default=30, help="Sequence length when using LSTM")
        parser.add_argument("--force-heuristic", action="store_true", help="Always use heuristic, skip loading models")

    def handle(self, *args, **options):
        seq_len = options.get("seq_len", 30)
        force_heuristic = options.get("force_heuristic", False)

        self.stdout.write("🔮 Running predictions for all airfields...")

        # Discover artifacts
        rf_th_path = TRAINED_DIR / "rf_thunder_calib.joblib"
        rf_gale_path = TRAINED_DIR / "rf_gale_calib.joblib"
        meta_th_path = TRAINED_DIR / "meta_thunder.joblib"
        meta_gale_path = TRAINED_DIR / "meta_gale.joblib"
        lstm_paths = [TRAINED_DIR / "lstm_thunder.keras", TRAINED_DIR / "lstm_thunder.h5"]
        cnn_paths = [TRAINED_DIR / "cnn_thunder.keras", TRAINED_DIR / "cnn_thunder.h5"]

        rf_th = rf_gale = meta_th = meta_gale = None
        lstm_th = cnn_th = None

        if not force_heuristic:
            if rf_th_path.exists():
                rf_th = safe_load_joblib(str(rf_th_path))
            if rf_gale_path.exists():
                rf_gale = safe_load_joblib(str(rf_gale_path))
            if meta_th_path.exists():
                meta_th = safe_load_joblib(str(meta_th_path))
            if meta_gale_path.exists():
                meta_gale = safe_load_joblib(str(meta_gale_path))

            # load keras models if present
            for p in lstm_paths:
                if p.exists():
                    lstm_th = safe_load_keras(str(p))
                    if lstm_th is not None:
                        break
            for p in cnn_paths:
                if p.exists():
                    cnn_th = safe_load_keras(str(p))
                    if cnn_th is not None:
                        break

        if force_heuristic or (not any([rf_th, rf_gale, meta_th, meta_gale, lstm_th, cnn_th])):
            self.stdout.write(self.style.NOTICE("No ensemble artifacts found (or forced heuristic). Falling back to heuristic predictor."))

        # For each airfield build features from most recent observations
        for af in Airfield.objects.all():
            obs_qs = WeatherObservation.objects.filter(station__airfield=af).order_by("-timestamp")[:max(30, seq_len)]
            if not obs_qs.exists():
                self.stdout.write(self.style.WARNING(f"No observations for {af.icao or af.name}, skipping"))
                continue

            obs = list(obs_qs)
            # tabular features
            tab = compute_tabular_from_obs(obs)
            # sequence for LSTM
            seq = build_sequence_from_obs(obs, seq_len=seq_len)

            # baseline heuristic
            temps = [o.temperature_c for o in obs if o.temperature_c is not None]
            hums = [o.humidity for o in obs if o.humidity is not None]
            gusts = [o.wind_gust_ms for o in obs if o.wind_gust_ms is not None]
            avg_temp = float(np.mean(temps)) if temps else None
            avg_hum = float(np.mean(hums)) if hums else None
            max_gust = float(max(gusts)) if gusts else 0.0

            heuristic_thunder = min(1.0, max(0.0, ((avg_hum or 50) - 50) / 50 + (0.02 * max_gust)))
            heuristic_gale = min(1.0, max(0.0, (max_gust / 40.0)))

            # ensemble predictions (start from None)
            p_rf_th = p_rf_g = p_lstm_th = p_cnn_th = None

            # RF predictions
            try:
                if rf_th is not None:
                    p_rf_th = float(rf_th.predict_proba(tab.reshape(1, -1))[:, 1])
                if rf_gale is not None:
                    p_rf_g = float(rf_gale.predict_proba(tab.reshape(1, -1))[:, 1])
                else:
                    p_rf_g = None
            except Exception as e:
                logger.warning("RF predict failed for %s: %s", af, e)
                p_rf_th = p_rf_g = None

            # LSTM
            try:
                if lstm_th is not None:
                    p_lstm_th = float(lstm_th.predict(seq.reshape(1, seq.shape[0], seq.shape[1])).ravel()[0])
            except Exception as e:
                logger.warning("LSTM predict failed for %s: %s", af, e)
                p_lstm_th = None

            # CNN -- build tiny synthetic image from last obs if model present
            try:
                if cnn_th is not None:
                    last = seq[-1]
                    hum = last[1]
                    gust = last[4]
                    img = np.zeros((32,32), dtype=np.float32)
                    cx, cy = 16, 16
                    sigma = 4.0
                    xs = np.arange(32)
                    ys = np.arange(32)[:,None]
                    g = np.exp(-((xs-cx)**2 + (ys-cy)**2)/(2*sigma**2))
                    intensity = min(1.0, max(0.0, (hum/100.0) * (1 + gust/30.0)))
                    img = (intensity * g)[...,None].astype(np.float32)
                    p_cnn_th = float(cnn_th.predict(img.reshape(1,32,32,1)).ravel()[0])
            except Exception as e:
                logger.warning("CNN predict failed for %s: %s", af, e)
                p_cnn_th = None

            # Meta learner: combine RF/LSTM/CNN probabilities if available
            final_th = None
            try:
                meta_feats = []
                for v in (p_rf_th, p_lstm_th, p_cnn_th):
                    meta_feats.append(float(v) if v is not None else float(heuristic_thunder))
                if meta_th is not None:
                    final_th = float(meta_th.predict_proba(np.array(meta_feats).reshape(1, -1))[:, 1])
                else:
                    vals = [x for x in [p_rf_th, p_lstm_th, p_cnn_th] if x is not None]
                    final_th = float(np.mean(vals)) if vals else float(heuristic_thunder)
            except Exception as e:
                logger.warning("Meta predict failed for %s: %s", af, e)
                final_th = float(heuristic_thunder)

            # Gale final (prefer RF or heuristic)
            final_gale = None
            try:
                if rf_gale is not None:
                    final_gale = float(p_rf_g) if p_rf_g is not None else float(heuristic_gale)
                else:
                    final_gale = float(heuristic_gale)
            except Exception as e:
                logger.warning("Gale final calc failed for %s: %s", af, e)
                final_gale = float(heuristic_gale)

            # clamp and round
            final_th = max(0.0, min(1.0, float(final_th)))
            final_gale = max(0.0, min(1.0, float(final_gale)))

            # Save Prediction (fixed: no duplicated kwargs)
            pred = Prediction.objects.create(
                airfield=af,
                thunderstorm_prob=round(final_th, 3),
                gale_wind_prob=round(final_gale, 3),
                confidence=1.0,
                created_at=timezone.now(),
                details={
                    "avg_temp": float(avg_temp) if avg_temp is not None else None,
                    "avg_humidity": float(avg_hum) if avg_hum is not None else None,
                    "max_gust": float(max_gust),
                    "src": {
                        "rf_th": round(float(p_rf_th), 4) if p_rf_th is not None else None,
                        "lstm_th": round(float(p_lstm_th), 4) if p_lstm_th is not None else None,
                        "cnn_th": round(float(p_cnn_th), 4) if p_cnn_th is not None else None,
                    }
                }
            )
            self.stdout.write(self.style.SUCCESS(f"Saved Prediction {pred.id} for {af.icao or af.name} (Thunder {pred.thunderstorm_prob:.3f}, Gale {pred.gale_wind_prob:.3f})"))
        self.stdout.write(self.style.SUCCESS("Done running predictions."))
