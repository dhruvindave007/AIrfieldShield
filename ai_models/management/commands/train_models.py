# ai_models/management/commands/train_models.py
import os
import random
import joblib
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TRAINED_DIR = "ai_models/trained"
os.makedirs(TRAINED_DIR, exist_ok=True)


def generate_synthetic_weather_data(n=5000):
    """
    Generate synthetic but realistic weather data with labels for thunderstorm and gale winds.
    """
    data = []
    labels_thunder = []
    labels_gale = []

    for _ in range(n):
        temp = random.uniform(20, 40)          # °C
        humidity = random.uniform(30, 95)      # %
        pressure = random.uniform(990, 1020)   # hPa
        wind_speed = random.uniform(0, 30)     # m/s
        gust = wind_speed + random.uniform(0, 15)

        # Synthetic rules to create labels
        thunderstorm = 1 if (humidity > 70 and temp > 28 and pressure < 1005) else 0
        gale = 1 if gust > 20 or wind_speed > 15 else 0

        data.append([temp, humidity, pressure, wind_speed, gust])
        labels_thunder.append(thunderstorm)
        labels_gale.append(gale)

    return np.array(data), np.array(labels_thunder), np.array(labels_gale)


class Command(BaseCommand):
    help = "Train AI models for thunderstorm and gale wind prediction using synthetic data"

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=5000, help="Number of synthetic samples to generate")

    def handle(self, *args, **options):
        n = options["count"]
        self.stdout.write(self.style.NOTICE(f"Generating {n} synthetic weather samples..."))
        X, y_thunder, y_gale = generate_synthetic_weather_data(n)

        # Split
        X_train, X_test, y_train_th, y_test_th = train_test_split(X, y_thunder, test_size=0.2, random_state=42)
        X_train2, X_test2, y_train_gale, y_test_gale = train_test_split(X, y_gale, test_size=0.2, random_state=42)

        # Train thunderstorm model
        model_th = RandomForestClassifier(n_estimators=100, random_state=42)
        model_th.fit(X_train, y_train_th)
        preds_th = model_th.predict(X_test)
        acc_th = accuracy_score(y_test_th, preds_th)

        # Train gale wind model
        model_gale = RandomForestClassifier(n_estimators=100, random_state=42)
        model_gale.fit(X_train2, y_train_gale)
        preds_gale = model_gale.predict(X_test2)
        acc_gale = accuracy_score(y_test_gale, preds_gale)

        # Save models
        joblib.dump(model_th, os.path.join(TRAINED_DIR, "thunderstorm_model.joblib"))
        joblib.dump(model_gale, os.path.join(TRAINED_DIR, "gale_model.joblib"))

        self.stdout.write(self.style.SUCCESS(f"✅ Training complete!"))
        self.stdout.write(self.style.SUCCESS(f"Thunderstorm model accuracy: {acc_th:.2f}"))
        self.stdout.write(self.style.SUCCESS(f"Gale wind model accuracy: {acc_gale:.2f}"))
        self.stdout.write(self.style.SUCCESS(f"Models saved in {TRAINED_DIR}/"))
