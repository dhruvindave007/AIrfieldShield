#!/usr/bin/env python3
"""
Management command: evaluate_models
Trains the ensemble models and outputs detailed accuracy metrics
"""
import os
import random
import joblib
import numpy as np
from pathlib import Path
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Where to save artifacts
BASE_DIR = Path("ai_models/trained")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def generate_tabular_and_sequences(n_samples=5000, seq_len=30, pos_frac=0.25):
    """
    Generate synthetic dataset with balanced classes
    """
    obs_dim = 5
    img_h, img_w = 32, 32

    X_tab = []
    X_seq = []
    X_img = []
    y_thunder = []
    y_gale = []

    # Generate sequences with controlled positive fraction
    pool = []
    for i in range(n_samples * 3):
        temps = np.random.normal(loc=30, scale=4, size=seq_len)
        hums = np.clip(np.random.normal(loc=60, scale=20, size=seq_len), 5, 100)
        press = np.random.normal(loc=1005, scale=6, size=seq_len)
        wind = np.abs(np.random.normal(loc=6, scale=4, size=seq_len))
        gust = wind + np.abs(np.random.normal(loc=2, scale=4, size=seq_len))

        seq = np.stack([temps, hums, press, wind, gust], axis=1)
        
        # Rule-based scores
        thunder_score = (seq[-1,1] - 60) / 40.0 + (seq[-1,0] - 28) / 8.0 + (1005 - seq[-1,2]) / 8.0
        gale_score = (seq[:,4].max() / 25.0) + (seq[:,3].mean() / 20.0)
        pool.append((seq.astype(np.float32), thunder_score, gale_score))

    pool_sorted = sorted(pool, key=lambda x: x[1], reverse=True)
    chosen = pool_sorted[:n_samples]
    
    for seq, thunder_score, gale_score in chosen:
        # Tabular features
        feats = []
        feats += [seq[:, 0].mean(), seq[:, 0].std(), seq[:, 0].max()]
        feats += [seq[:, 1].mean(), seq[:, 1].std()]
        feats += [seq[:, 2].mean(), seq[:, 2].std()]
        feats += [seq[:, 3].mean(), seq[:, 3].max()]
        feats += [seq[:, 4].max()]

        # Synthetic radar image
        img = np.zeros((img_h, img_w), dtype=np.float32)
        n_blobs = random.randint(1, 4)
        for b in range(n_blobs):
            cx = random.uniform(6, img_w - 6)
            cy = random.uniform(6, img_h - 6)
            sigma = random.uniform(2.0, 6.0)
            intensity = np.clip((seq[-1,1] / 100.0) * (1 + seq[-1,4] / 30.0) + random.uniform(-0.15, 0.15), 0, 1)
            xs = np.arange(img_w)
            ys = np.arange(img_h)[:, None]
            g = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2*sigma**2))
            img += intensity * g
        img = img / (img.max() + 1e-8)
        img = np.clip(img, 0, 1)

        # Labels with noise
        thunder_label = 1 if (thunder_score > 0.3) else 0
        gale_label = 1 if (gale_score > 0.9) else 0

        if random.random() < 0.08:
            thunder_label = 1 - thunder_label
        if random.random() < 0.05:
            gale_label = 1 - gale_label

        X_tab.append(np.array(feats, dtype=np.float32))
        X_seq.append(seq)
        X_img.append(img[..., None].astype(np.float32))
        y_thunder.append(thunder_label)
        y_gale.append(gale_label)

    return (np.stack(X_tab), np.stack(X_seq), np.stack(X_img),
            np.array(y_thunder, dtype=np.int32), np.array(y_gale, dtype=np.int32))


def build_lstm_model(seq_len=30, obs_dim=5):
    inp = layers.Input(shape=(seq_len, obs_dim), name="lstm_input")
    x = layers.Masking()(inp)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="thunder_out")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="AUC")])
    return model


def build_cnn_model(h=32, w=32):
    inp = layers.Input(shape=(h, w, 1), name="cnn_input")
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="cnn_out")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="AUC")])
    return model


def evaluate_binary_model(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics for binary classification"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics


class Command(BaseCommand):
    help = "Train models and display detailed accuracy metrics"

    def add_arguments(self, parser):
        parser.add_argument("--samples", type=int, default=8000, help="Number of synthetic samples")
        parser.add_argument("--seq-len", type=int, default=30, help="Sequence length for LSTM")
        parser.add_argument("--epochs", type=int, default=10, help="Epochs for LSTM/CNN training")
        parser.add_argument("--batch", type=int, default=64, help="Batch size")

    def handle(self, *args, **options):
        n = int(options["samples"])
        seq_len = int(options["seq_len"])
        epochs = int(options["epochs"])
        batch = int(options["batch"])

        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write(self.style.SUCCESS("         AIRFIELD SHIELD - MODEL TRAINING & EVALUATION"))
        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write("")
        
        self.stdout.write(self.style.NOTICE(f"📊 Generating {n} synthetic samples (seq_len={seq_len})..."))
        X_tab, X_seq, X_img, y_th, y_gale = generate_tabular_and_sequences(n, seq_len)

        # Data split: Train (60%), Validation (20%), Test (20%)
        # First split: 80% train+val, 20% test
        (X_tab_trainval, X_tab_te,
         X_seq_trainval, X_seq_te,
         X_img_trainval, X_img_te,
         y_trainval_th, y_te_th) = train_test_split(
            X_tab, X_seq, X_img, y_th, test_size=0.2, random_state=42)

        _, _, _, _, _, _, y_trainval_gale, y_te_gale = train_test_split(
            X_tab, X_seq, X_img, y_gale, test_size=0.2, random_state=42)

        # Second split: split train+val into train (75% of 80% = 60%) and val (25% of 80% = 20%)
        (X_tab_tr, X_tab_val,
         X_seq_tr, X_seq_val,
         X_img_tr, X_img_val,
         y_tr_th, y_val_th) = train_test_split(
            X_tab_trainval, X_seq_trainval, X_img_trainval, y_trainval_th, 
            test_size=0.25, random_state=42)

        _, _, _, _, _, _, y_tr_gale, y_val_gale = train_test_split(
            X_tab_trainval, X_seq_trainval, X_img_trainval, y_trainval_gale, 
            test_size=0.25, random_state=42)

        self.stdout.write(f"  Training set: {len(X_tab_tr)} samples")
        self.stdout.write(f"  Validation set: {len(X_tab_val)} samples")
        self.stdout.write(f"  Test set: {len(X_tab_te)} samples")
        self.stdout.write("")

        # Store metrics
        all_metrics = {}

        # ================ Random Forest - Thunderstorm ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🌲 Training Random Forest (Tabular) - Thunderstorm Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        rf_th = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
        rf_th.fit(X_tab_tr, y_tr_th)
        calib_rf_th = CalibratedClassifierCV(rf_th, cv=5)
        calib_rf_th.fit(X_tab_tr, y_tr_th)
        p_rf_th = calib_rf_th.predict_proba(X_tab_te)[:, 1]
        
        metrics_rf_th = evaluate_binary_model(y_te_th, p_rf_th)
        all_metrics['rf_thunderstorm'] = metrics_rf_th
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_rf_th['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_rf_th['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_rf_th['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_rf_th['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_rf_th['auc']:.4f}")
        self.stdout.write("")

        joblib.dump(calib_rf_th, str(BASE_DIR / "rf_thunder_calib.joblib"))

        # ================ Random Forest - Gale Wind ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🌲 Training Random Forest (Tabular) - Gale Wind Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        rf_gale = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
        rf_gale.fit(X_tab_tr, y_tr_gale)
        calib_rf_g = CalibratedClassifierCV(rf_gale, cv=5)
        calib_rf_g.fit(X_tab_tr, y_tr_gale)
        p_rf_g = calib_rf_g.predict_proba(X_tab_te)[:, 1]
        
        metrics_rf_gale = evaluate_binary_model(y_te_gale, p_rf_g)
        all_metrics['rf_gale'] = metrics_rf_gale
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_rf_gale['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_rf_gale['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_rf_gale['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_rf_gale['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_rf_gale['auc']:.4f}")
        self.stdout.write("")

        joblib.dump(calib_rf_g, str(BASE_DIR / "rf_gale_calib.joblib"))
        joblib.dump(rf_th, str(BASE_DIR / "rf_thunder_base.joblib"))
        joblib.dump(rf_gale, str(BASE_DIR / "rf_gale_base.joblib"))

        # ================ LSTM - Thunderstorm ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🔗 Training LSTM (Sequence) - Thunderstorm Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        lstm_th = build_lstm_model(seq_len=seq_len, obs_dim=X_seq.shape[2])
        es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        lstm_th.fit(X_seq_tr, y_tr_th, validation_data=(X_seq_val, y_val_th),
                    epochs=epochs, batch_size=batch, callbacks=[es], verbose=0)
        p_lstm_th = lstm_th.predict(X_seq_te, verbose=0).ravel()
        
        metrics_lstm = evaluate_binary_model(y_te_th, p_lstm_th)
        all_metrics['lstm_thunderstorm'] = metrics_lstm
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_lstm['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_lstm['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_lstm['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_lstm['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_lstm['auc']:.4f}")
        self.stdout.write("")

        lstm_th.save(str(BASE_DIR / "lstm_thunder.keras"))

        # ================ CNN - Thunderstorm ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🖼️  Training CNN (Radar Image) - Thunderstorm Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        cnn_th = build_cnn_model(h=X_img.shape[1], w=X_img.shape[2])
        es2 = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        cnn_th.fit(X_img_tr, y_tr_th, validation_data=(X_img_val, y_val_th),
                   epochs=epochs, batch_size=batch, callbacks=[es2], verbose=0)
        p_cnn_th = cnn_th.predict(X_img_te, verbose=0).ravel()
        
        metrics_cnn = evaluate_binary_model(y_te_th, p_cnn_th)
        all_metrics['cnn_thunderstorm'] = metrics_cnn
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_cnn['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_cnn['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_cnn['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_cnn['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_cnn['auc']:.4f}")
        self.stdout.write("")

        cnn_th.save(str(BASE_DIR / "cnn_thunder.keras"))

        # ================ Meta-Learner - Thunderstorm ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🎯 Training Meta-Learner (Ensemble) - Thunderstorm Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        # Train meta-learner on VALIDATION set predictions to avoid data leakage
        rf_val_th = calib_rf_th.predict_proba(X_tab_val)[:, 1]
        lstm_val_th = lstm_th.predict(X_seq_val, verbose=0).ravel()
        cnn_val_th = cnn_th.predict(X_img_val, verbose=0).ravel()
        meta_X_train = np.vstack([rf_val_th, lstm_val_th, cnn_val_th]).T
        meta_y_train = y_val_th
        
        meta = LogisticRegression(max_iter=2000)
        meta.fit(meta_X_train, meta_y_train)
        
        # Evaluate meta-learner on TEST set predictions
        rf_te_th = calib_rf_th.predict_proba(X_tab_te)[:, 1]
        lstm_te_th = lstm_th.predict(X_seq_te, verbose=0).ravel()
        cnn_te_th = cnn_th.predict(X_img_te, verbose=0).ravel()
        meta_X_test = np.vstack([rf_te_th, lstm_te_th, cnn_te_th]).T
        meta_p = meta.predict_proba(meta_X_test)[:, 1]
        
        metrics_meta = evaluate_binary_model(y_te_th, meta_p)
        all_metrics['meta_thunderstorm'] = metrics_meta
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_meta['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_meta['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_meta['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_meta['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_meta['auc']:.4f}")
        self.stdout.write("")

        joblib.dump(meta, str(BASE_DIR / "meta_thunder.joblib"))

        # ================ Meta-Learner - Gale ================
        self.stdout.write(self.style.SUCCESS("─" * 80))
        self.stdout.write(self.style.NOTICE("🎯 Training Meta-Learner - Gale Wind Prediction"))
        self.stdout.write(self.style.SUCCESS("─" * 80))
        
        # Train meta-learner on VALIDATION set predictions to avoid data leakage
        p_rf_g_val = calib_rf_g.predict_proba(X_tab_val)[:, 1]
        meta_gale = LogisticRegression(max_iter=1000)
        meta_gale.fit(p_rf_g_val.reshape(-1, 1), y_val_gale)
        
        # Evaluate on test set
        p_rf_g_test = calib_rf_g.predict_proba(X_tab_te)[:, 1]
        meta_gale_p = meta_gale.predict_proba(p_rf_g_test.reshape(-1, 1))[:, 1]
        metrics_meta_gale = evaluate_binary_model(y_te_gale, meta_gale_p)
        all_metrics['meta_gale'] = metrics_meta_gale
        
        self.stdout.write(f"  ✓ Accuracy:  {metrics_meta_gale['accuracy']:.4f}")
        self.stdout.write(f"  ✓ Precision: {metrics_meta_gale['precision']:.4f}")
        self.stdout.write(f"  ✓ Recall:    {metrics_meta_gale['recall']:.4f}")
        self.stdout.write(f"  ✓ F1 Score:  {metrics_meta_gale['f1']:.4f}")
        self.stdout.write(f"  ✓ ROC-AUC:   {metrics_meta_gale['auc']:.4f}")
        self.stdout.write("")

        joblib.dump(meta_gale, str(BASE_DIR / "meta_gale.joblib"))

        # ================ Summary ================
        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write(self.style.SUCCESS("                        📊 FINAL SUMMARY"))
        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write("")
        
        self.stdout.write("Model Performance (Test Set):")
        self.stdout.write("")
        
        for model_name, metrics in all_metrics.items():
            self.stdout.write(f"  {model_name.replace('_', ' ').title()}:")
            self.stdout.write(f"    • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            self.stdout.write(f"    • Precision: {metrics['precision']:.4f}")
            self.stdout.write(f"    • Recall:    {metrics['recall']:.4f}")
            self.stdout.write(f"    • F1 Score:  {metrics['f1']:.4f}")
            self.stdout.write(f"    • ROC-AUC:   {metrics['auc']:.4f}")
            self.stdout.write("")

        self.stdout.write(self.style.SUCCESS("✅ All models trained and saved to: " + str(BASE_DIR)))
        self.stdout.write("")
        self.stdout.write("Saved models:")
        self.stdout.write("  • rf_thunder_calib.joblib")
        self.stdout.write("  • rf_gale_calib.joblib")
        self.stdout.write("  • lstm_thunder.keras")
        self.stdout.write("  • cnn_thunder.keras")
        self.stdout.write("  • meta_thunder.joblib")
        self.stdout.write("  • meta_gale.joblib")
        self.stdout.write(self.style.SUCCESS("=" * 80))
