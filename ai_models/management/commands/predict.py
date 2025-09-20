# ai_models/management/commands/predict.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import Airfield, WeatherObservation, Prediction
import numpy as np
import math
import joblib
from pathlib import Path
import tensorflow as tf

TRAINED_DIR = Path("ai_models/trained")

# helper to safely load a Keras model saved in modern format
def load_keras_model(path):
    if not Path(path).exists():
        return None
    # no custom objects expected now
    return tf.keras.models.load_model(str(path))

class Command(BaseCommand):
    help = "Run ensemble predictions for all airfields using trained artifacts"

    def add_arguments(self, parser):
        parser.add_argument("--force-sim", action="store_true", help="If no observations, simulate predictions from fake aggregated values")

    def handle(self, *args, **options):
        self.stdout.write("🔮 Running ensemble predictions...")

        # load artifacts (gracefully)
        rf_th_calib = None
        rf_g_calib = None
        meta_th = None
        meta_g = None
        lstm_th = None
        cnn_th = None

        try:
            rf_th_calib = joblib.load(str(TRAINED_DIR / "rf_thunder_calib.joblib"))
            rf_g_calib = joblib.load(str(TRAINED_DIR / "rf_gale_calib.joblib"))
            meta_th = joblib.load(str(TRAINED_DIR / "meta_thunder.joblib"))
            meta_g = joblib.load(str(TRAINED_DIR / "meta_gale.joblib"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Warning: failed to load some joblib artifacts: {e}"))

        # load keras models using modern API
        try:
            lstm_th = load_keras_model(TRAINED_DIR / "lstm_thunder.keras")
            cnn_th = load_keras_model(TRAINED_DIR / "cnn_thunder.keras")
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Warning: failed to load Keras models: {e}"))

        for af in Airfield.objects.all():
            # get recent observations for this airfield
            obs_qs = WeatherObservation.objects.filter(station__airfield=af).order_by("-timestamp")[:30]
            if not obs_qs.exists():
                if not options.get("force_sim"):
                    self.stdout.write(self.style.WARNING(f"No observations for {af.icao}, skipping"))
                    continue
                # simulate an aggregate if forced
                temps = [30.0]; hums = [70.0]; gusts=[8.0]
            else:
                temps = [o.temperature_c for o in obs_qs if o.temperature_c is not None]
                hums = [o.humidity for o in obs_qs if o.humidity is not None]
                gusts = [o.wind_gust_ms for o in obs_qs if o.wind_gust_ms is not None]

            avg_temp = float(np.mean(temps)) if temps else None
            avg_hum = float(np.mean(hums)) if hums else None
            max_gust = float(max(gusts)) if gusts else 0.0

            # Build tabular feature used by RF (must match training aggregates)
            # temp mean, temp std, temp max, hum mean, hum std, press mean, press std, wind mean, wind max, peak gust
            # Some fields (pressure) may not be available from obs, use reasonable defaults
            press_vals = [o.pressure_hpa for o in obs_qs if getattr(o, "pressure_hpa", None) is not None]
            press_mean = float(np.mean(press_vals)) if press_vals else 1005.0
            press_std = float(np.std(press_vals)) if press_vals else 4.0
            temp_std = float(np.std(temps)) if temps else 2.0
            wind_mean = float(np.mean([o.wind_speed_ms for o in obs_qs if getattr(o, "wind_speed_ms", None) is not None])) if obs_qs else 5.0
            wind_max = float(np.max([o.wind_speed_ms for o in obs_qs if getattr(o, "wind_speed_ms", None) is not None])) if obs_qs else 8.0

            tab_feat = np.array([
                avg_temp or 30.0,
                temp_std,
                (max(temps) if temps else (avg_temp or 30.0)),
                avg_hum or 60.0,
                float(np.std(hums)) if hums else 8.0,
                press_mean,
                press_std,
                wind_mean,
                wind_max,
                max_gust
            ], dtype=np.float32).reshape(1, -1)

            # RF probabilities (calibrated) if available
            p_rf_th = rf_th_calib.predict_proba(tab_feat)[:, 1] if rf_th_calib is not None else np.array([0.0])
            p_rf_g = rf_g_calib.predict_proba(tab_feat)[:, 1] if rf_g_calib is not None else np.array([0.0])

            # LSTM / CNN inputs construction (use last 30 obs or padded)
            seq_input = None
            img_input = None
            if obs_qs.exists():
                # build seq of length 30, pad/truncate using most recent obs (descending order)
                last_obs = list(obs_qs)[0:30]  # newest first
                last_obs_rev = list(reversed(last_obs))  # oldest -> newest
                seq = []
                for o in last_obs_rev:
                    seq.append([
                        getattr(o, "temperature_c", 30.0) or 30.0,
                        getattr(o, "humidity", 60.0) or 60.0,
                        getattr(o, "pressure_hpa", 1005.0) or 1005.0,
                        getattr(o, "wind_speed_ms", 5.0) or 5.0,
                        getattr(o, "wind_gust_ms", 5.0) or 5.0,
                    ])
                # pad if needed at front with last-known values
                while len(seq) < 30:
                    seq.insert(0, seq[0] if seq else [30, 60, 1005, 5, 5])
                seq_input = np.array(seq[-30:], dtype=np.float32).reshape(1, 30, 5)

                # create tiny synthetic image from recent humidity/gust to feed CNN (simple radial intensity)
                last_h = seq[-1][1]
                last_g = seq[-1][4]
                img = np.zeros((32, 32), dtype=np.float32)
                cx, cy = 16, 16
                sigma = 4.0 + (last_g / 10.0)
                xs = np.arange(32)
                ys = np.arange(32)[:, None]
                g = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2*sigma**2))
                img += (last_h / 100.0) * g
                img = img / (img.max() + 1e-8)
                img_input = img.reshape(1, 32, 32, 1).astype(np.float32)

            # get predictions from three components (use fallback zeros)
            rf_prob = float(p_rf_th[0]) if p_rf_th is not None else 0.0
            lstm_prob = float(lstm_th.predict(seq_input).ravel()[0]) if (lstm_th is not None and seq_input is not None) else 0.0
            cnn_prob = float(cnn_th.predict(img_input).ravel()[0]) if (cnn_th is not None and img_input is not None) else 0.0

            # assemble meta features and predict meta probability (thunder)
            meta_feat = np.array([[rf_prob, lstm_prob, cnn_prob]])
            prob_th = meta_th.predict_proba(meta_feat)[:, 1][0] if meta_th is not None else rf_prob * 0.6 + lstm_prob * 0.2 + cnn_prob * 0.2

            # gale prediction via RF calibrated + small meta if available
            p_rf_g_val = float(p_rf_g[0]) if p_rf_g is not None else 0.0
            try:
                prob_g = meta_g.predict_proba(np.array([[p_rf_g_val]]))[:, 1][0] if meta_g is not None else p_rf_g_val
            except Exception:
                prob_g = p_rf_g_val

            # confidence: combine model AUC proxies / fallback to 0.7
            # here we use a simple heuristic: if any model contributed strongly, confidence up
            conf = 0.6 + min(0.4, abs(prob_th - 0.5) * 0.8)  # heuristic scale 0.6..1.0
            conf = round(float(min(1.0, max(0.0, conf))), 2)

            # Round probabilities to 3 decimals
            prob_th = float(round(prob_th, 3))
            prob_g = float(round(prob_g, 3))

            # Save Prediction object
            pred = Prediction.objects.create(
                airfield=af,
                thunderstorm_prob=prob_th,
                gale_wind_prob=prob_g,
                confidence=conf,
                details={
                    "avg_temp": float(avg_temp) if avg_temp is not None else None,
                    "avg_humidity": float(avg_hum) if avg_hum is not None else None,
                    "avg_pressure": float(press_mean),
                    "max_gust": float(max_gust),
                    "rf_prob_th": float(round(rf_prob, 3)),
                    "lstm_prob_th": float(round(lstm_prob, 3)),
                    "cnn_prob_th": float(round(cnn_prob, 3)),
                    "rf_prob_g": float(round(p_rf_g_val, 3)),
                },
            )

            self.stdout.write(self.style.SUCCESS(
                f"Saved Prediction {pred.id} for {af.icao} (thunder={pred.thunderstorm_prob}, gale={pred.gale_wind_prob}, conf={pred.confidence})"
            ))

        self.stdout.write(self.style.SUCCESS("✅ All done."))
