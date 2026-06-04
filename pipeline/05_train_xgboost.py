"""
pipeline/train_xgboost.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Trains an XGBoost classifier for next-day rain prediction.

Input  : data/processed/azamgarh_weather_final.csv
Output : models/xgboost/rain_model.json
         models/xgboost/model_meta.json

Usage:
    python pipeline/train_xgboost.py
"""

# ──────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────
import os
import sys
import json
import warnings

# ──────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)
from sklearn.utils.class_weight import (
    compute_sample_weight,
)

# ──────────────────────────────────────────────────────────
# Project Configuration
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.config import (
    DATA_PROCESSED_DIR,
    MODEL_XGBOOST_DIR,
    XGBOOST_MODEL_FILE,
    XGBOOST_META_FILE,
)

# ──────────────────────────────────────────────────────────
# Reproducibility Configuration
# ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    DATA_PROCESSED_DIR,
    "azamgarh_weather_final.csv"
)

MODEL_PATH = os.path.join(
    MODEL_XGBOOST_DIR,
    XGBOOST_MODEL_FILE
)

META_PATH = os.path.join(
    MODEL_XGBOOST_DIR,
    XGBOOST_META_FILE
)

# Create model directory if it does not exist
os.makedirs(
    MODEL_XGBOOST_DIR,
    exist_ok=True
)

# ──────────────────────────────────────────────────────────
# Feature Configuration
# ──────────────────────────────────────────────────────────
# Input features used by the XGBoost rain prediction model.
# Must match exactly with:
#   • feature_engineering.py output
#   • API inference pipeline
#   • model metadata (model_meta.json)
# Any change here requires model retraining.
FEATURES = [
    "temperature",
    "surface_pressure",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "precipitation",
    "humidity",         
    "wind_speed",
    "temp_rolling_6",
    "temp_lag_1",
    "temp_lag_24",
    "month",
]

# Target variable:
# 1 → Rain expected tomorrow
# 0 → No rain expected tomorrow
TARGET = "rain_tomorrow"

# ──────────────────────────────────────────────────────────
# XGBoost Training Configuration/Hyperparameters
# ──────────────────────────────────────────────────────────
# Hyperparameters selected after experimentation.
# Optimized for:
#   • Class imbalance handling
#   • Time-series weather forecasting
#   • Good ROC-AUC and F1 performance
XGB_PARAMS = dict(
    n_estimators          = 300,
    max_depth             = 5,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    use_label_encoder     = False,
    eval_metric           = "logloss",
    early_stopping_rounds = 20,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
)


# ──────────────────────────────────────────────────────────
# Threshold Optimization Utility
# ──────────────────────────────────────────────────────────
# Searches probability thresholds from 0.10–0.90
# and returns the threshold that maximizes F1-score.
#
# Why?
# Default threshold (0.50) is not always optimal for
# imbalanced rainfall datasets.
# Tuning often improves recall and overall F1 score.
# ──────────────────────────────────────────────────────────
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep thresholds 0.10 → 0.90 and return the one that maximises F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.91, 0.05):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return round(float(best_thr), 2)


# ──────────────────────────────────────────────────────────
# Rain Prediction Utility(used by app/services/ml_services.py)
# ──────────────────────────────────────────────────────────
# Used by FastAPI inference service after training.
#
# Converts raw feature dictionary into a DataFrame,
# predicts rain probability,
# applies classification threshold,
# and generates a confidence label.
# ──────────────────────────────────────────────────────────
def predict_rain(
    model: XGBClassifier,
    input_dict: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Predict rain tomorrow from a feature dictionary.

    Args:
        model      : trained XGBClassifier
        input_dict : dict with keys matching FEATURES list
        threshold  : classification threshold (use tuned best_threshold from meta)

    Returns:
        dict — rain_tomorrow (bool), probability (float), confidence (str)

    Example:
        result = predict_rain(model, {
            "temperature": 28.5, "surface_pressure": 1005.0,
            "total_cloud_cover": 0.6, "low_cloud_cover": 0.3,
            "medium_cloud_cover": 0.2, "high_cloud_cover": 0.1,
            "precipitation": 0.0, "humidity": 75.0, "wind_speed": 3.5,
            "temp_rolling_6": 27.8, "temp_lag_1": 27.2,
            "temp_lag_24": 26.5, "month": 7,
        })
    """
    row  = pd.DataFrame([{feat: input_dict[feat] for feat in FEATURES}])
    prob = float(model.predict_proba(row)[0][1])

    if prob > 0.75:
        confidence = "high (rain likely)"
    elif prob > 0.55:
        confidence = "medium"
    elif prob < 0.25:
        confidence = "high (no rain likely)"
    else:
        confidence = "low (uncertain)"

    return {
        "rain_tomorrow"  : bool(prob >= threshold),
        "probability"    : round(prob, 4),
        "confidence"     : confidence,
        "threshold_used" : threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRAIN_XGBOOST.PY — Rain Prediction Classifier")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print(f"\n[1/8] Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X  = df[FEATURES]
    y  = df[TARGET].astype(int)

    print(f"  Dataset shape : {df.shape}")
    print(f"  Rain=1        : {y.sum():,}  ({100*y.mean():.1f}%)")
    print(f"  Rain=0        : {(~y.astype(bool)).sum():,}  ({100*(1-y.mean()):.1f}%)")
    print(f"  Imbalance     : {(~y.astype(bool)).sum() / y.sum():.1f}:1")

    # ── 2. Time-based train/test split ─────────────────────────────────────────
    print("\n[2/8] Splitting train / test (80/20, time-ordered, no shuffle)...")
    split_idx       = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    train_dates = df["valid_time"].iloc[:split_idx]
    test_dates  = df["valid_time"].iloc[split_idx:]
    print(f"  Train : {len(X_train):,} rows  ({train_dates.min().date()} → {train_dates.max().date()})")
    print(f"  Test  : {len(X_test):,}  rows  ({test_dates.min().date()}  → {test_dates.max().date()})")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"  Sample weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    # ── 3. Cross-validation ────────────────────────────────────────────────────
    print("\n[3/8] 5-fold TimeSeriesSplit cross-validation...")
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = {"accuracy": [], "f1": [], "roc_auc": []}

    print(f"\n  {'Fold':<6} {'Train':>8} {'Val':>8} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print(f"  {'─'*55}")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        sw = compute_sample_weight("balanced", y=ytr)

        # CV uses no early_stopping (no eval_set)
        m = XGBClassifier(**{k: v for k, v in XGB_PARAMS.items()
                             if k != "early_stopping_rounds"})
        m.fit(Xtr, ytr, sample_weight=sw, verbose=False)

        yp  = m.predict(Xval)
        ypp = m.predict_proba(Xval)[:, 1]

        acc = accuracy_score(yval, yp)
        f1  = f1_score(yval, yp, zero_division=0)
        auc = roc_auc_score(yval, ypp)

        cv_scores["accuracy"].append(acc)
        cv_scores["f1"].append(f1)
        cv_scores["roc_auc"].append(auc)
        print(f"  {fold:<6} {len(Xtr):>8,} {len(Xval):>8,} {acc:>10.4f} {f1:>8.4f} {auc:>10.4f}")

    print(f"  {'─'*55}")
    print(f"  {'Mean':<6} {'':>8} {'':>8} "
          f"{np.mean(cv_scores['accuracy']):>10.4f} "
          f"{np.mean(cv_scores['f1']):>8.4f} "
          f"{np.mean(cv_scores['roc_auc']):>10.4f}")
    print(f"  {'Std':<6} {'':>8} {'':>8} "
          f"{np.std(cv_scores['accuracy']):>10.4f} "
          f"{np.std(cv_scores['f1']):>8.4f} "
          f"{np.std(cv_scores['roc_auc']):>10.4f}")

    # ── 4. Final model ─────────────────────────────────────────────────────────
    print("\n[4/8] Training final model on full training set...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_test, y_test)],
        verbose       = 50,
    )
    print(f"\n  Best iteration: {model.best_iteration}")

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    print("\n[5/8] Evaluating on test set (threshold = 0.5)...")
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)
    ap  = average_precision_score(y_test, y_pred_prob)

    print(f"\n  {'═'*45}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Avg Precision : {ap:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Rain", "Rain"],
                                zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':>16} Pred No Rain  Pred Rain")
    print(f"  True No Rain  {cm[0][0]:>12}  {cm[0][1]:>9}")
    print(f"  True Rain     {cm[1][0]:>12}  {cm[1][1]:>9}")

    # ── 6. Feature importance ──────────────────────────────────────────────────
    print("\n[6/8] Feature importance (by Gain)...")
    importance = model.get_booster().get_score(importance_type="gain")
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  {'Feature':<22} {'Gain':>8}  Bar")
    print(f"  {'─'*55}")
    max_gain = max(importance.values())
    for feat, score in importance.items():
        bar = "█" * int(score / max_gain * 30)
        print(f"  {feat:<22} {score:>8.2f}  {bar}")

    # ── 7. Threshold tuning ────────────────────────────────────────────────────
    print("\n[7/8] Tuning classification threshold...")
    best_thr     = find_best_threshold(y_test.values, y_pred_prob)
    y_pred_tuned = (y_pred_prob >= best_thr).astype(int)
    f1_tuned     = f1_score(y_test, y_pred_tuned, zero_division=0)
    acc_tuned    = accuracy_score(y_test, y_pred_tuned)

    print(f"  Best threshold : {best_thr}  (maximises F1)")
    print(f"  F1  @ {best_thr}    : {f1_tuned:.4f}  (was {f1:.4f} @ 0.5)")
    print(f"  Acc @ {best_thr}    : {acc_tuned:.4f}  (was {acc:.4f} @ 0.5)")

    # ── 8. Save ────────────────────────────────────────────────────────────────
    print("\n[8/8] Saving model and metadata...")
    model.save_model(MODEL_PATH)
    print(f"  Model saved    → {MODEL_PATH}")

    meta = {
        "model"          : "XGBoostClassifier",
        "features"       : FEATURES,
        "target"         : TARGET,
        "best_threshold" : best_thr,
        "best_iteration" : int(model.best_iteration),
        "train_rows"     : int(len(X_train)),
        "test_rows"      : int(len(X_test)),
        "hyperparameters": {k: v for k, v in XGB_PARAMS.items()},
        "cv_results": {
            "mean_accuracy" : round(float(np.mean(cv_scores["accuracy"])), 4),
            "std_accuracy"  : round(float(np.std(cv_scores["accuracy"])),  4),
            "mean_f1"       : round(float(np.mean(cv_scores["f1"])),       4),
            "std_f1"        : round(float(np.std(cv_scores["f1"])),        4),
            "mean_roc_auc"  : round(float(np.mean(cv_scores["roc_auc"])),  4),
            "std_roc_auc"   : round(float(np.std(cv_scores["roc_auc"])),   4),
        },
        
        "roc_auc"       : round(float(auc), 4),
        "avg_precision" : round(float(ap),  4),

        "metrics_at_0_5": {
            "accuracy"      : round(float(acc), 4),
            "f1_score"      : round(float(f1),  4),

        },
        "metrics_at_best_threshold": {
            "threshold" : best_thr,
            "accuracy"  : round(float(acc_tuned), 4),
            "f1_score"  : round(float(f1_tuned),  4),
        },
        "feature_importance_gain": {k: round(v, 4) for k, v in importance.items()},
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ XGBoost training complete")
    print(f"     ROC-AUC    : {auc:.4f}")
    print(f"     F1 (tuned) : {f1_tuned:.4f}  @ threshold {best_thr}")
    print(f"     CV ROC-AUC : {np.mean(cv_scores['roc_auc']):.4f}"
          f" ± {np.std(cv_scores['roc_auc']):.4f}")
    print(f"{'=' * 60}")

    # ── Quick demo ─────────────────────────────────────────────────────────────
    sample = X_test.iloc[0].to_dict()
    result = predict_rain(model, sample, threshold=best_thr)
    actual = int(y_test.iloc[0])
    print(f"\n  Demo prediction:")
    print(f"    Input      : {sample}")
    print(f"    Prediction : {'🌧 Rain' if result['rain_tomorrow'] else '☀ No Rain'}")
    print(f"    Probability: {result['probability']}")
    print(f"    Confidence : {result['confidence']}")
    print(f"    Actual     : {'Rain' if actual else 'No Rain'}")

    return model, meta


if __name__ == "__main__":
    main()