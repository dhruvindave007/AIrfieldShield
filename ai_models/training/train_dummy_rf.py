#!/usr/bin/env python3
"""
Train a dummy RandomForest classifier on synthetic weather features and save it to
<project_root>/models/rf_model.joblib so the predict management command can load it.

Save path (relative to this file): ../../models/rf_model.joblib  -> resolves to <project_root>/models/rf_model.joblib
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RANDOM_STATE = 42


def synthesize_data(n=20000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    # Features consistent with extract_features_for_airfield:
    # mean_temp, mean_humidity, mean_pressure, pressure_delta, max_gust, mean_wind_speed
    mean_temp = rng.uniform(0, 40, size=n)  # °C
    mean_humidity = rng.uniform(10, 100, size=n)  # %
    mean_pressure = rng.uniform(980, 1035, size=n)  # hPa
    pressure_delta = rng.normal(loc=0.0, scale=2.0, size=n)  # hPa change over window
    max_gust = rng.uniform(0, 30, size=n)  # m/s
    mean_wind_speed = rng.uniform(0, 20, size=n)  # m/s

    df = pd.DataFrame({
        "mean_temp": mean_temp,
        "mean_humidity": mean_humidity,
        "mean_pressure": mean_pressure,
        "pressure_delta": pressure_delta,
        "max_gust": max_gust,
        "mean_wind_speed": mean_wind_speed,
    })

    # Create a synthetic "severe" target (1 = severe weather event likely)
    # The rule increases severity when gusts are high, pressure low or dropping fast + high humidity.
    score = (
        (max_gust / 30.0) * 1.5  # gust contribution
        + (1.0 - (mean_pressure - 980) / (1035 - 980)) * 0.8  # lower pressure -> more risk
        + (np.tanh(np.maximum(0, pressure_delta) / 2.0)) * 0.6  # rapidly falling pressure -> risk
        + ((mean_humidity - 50) / 50.0).clip(0, 1.0) * 0.4  # high humidity raises risk
    )

    # Add some noise and convert to probability-like
    prob = 1 / (1 + np.exp(- (score * 2.0 - 1.0)))  # sigmoid to map to (0,1)
    # The threshold to label 'severe' can be tuned; choose 0.5 but inject randomness for realism
    target = (prob + rng.normal(0, 0.1, size=n)) > 0.5
    df["severe"] = target.astype(int)

    return df


def train_and_save(df, model_path, random_state=RANDOM_STATE):
    features = ["mean_temp", "mean_humidity", "mean_pressure", "pressure_delta", "max_gust", "mean_wind_speed"]
    X = df[features].values
    y = df["severe"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    print("Training RandomForestClassifier on synthetic data...")
    clf.fit(X_train, y_train)

    print("Evaluating on test split...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        print("ROC AUC could not be computed.")

    # Ensure model dir exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved model to: {model_path}")
    # Save a small metadata file with feature order to help debugging
    metadata = {"feature_order": features, "model_type": "RandomForestClassifier", "random_state": random_state}
    joblib.dump(metadata, model_path.with_suffix(".meta.joblib"))
    print(f"Saved metadata to: {model_path.with_suffix('.meta.joblib')}")


if __name__ == "__main__":
    # Resolve project root relative to this file:
    # file: <project_root>/ai_models/training/train_dummy_rf.py
    # project_root = parents[2]
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models"
    model_file = model_dir / "rf_model.joblib"

    print(f"Project root inferred as: {project_root}")
    print("Synthesizing dataset...")
    df = synthesize_data(n=20000)
    train_and_save(df, model_file)
    print("\nDONE.\n")
    print("Next: run the predict command to use this model:")
    print(f"  python manage.py predict --model-path {model_file}")
