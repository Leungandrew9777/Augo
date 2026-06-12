#!/usr/bin/env python3
"""
Step 3: train XGBoost 3-class model on international features.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(APP_DIR, "international_with_features.csv")
MODEL_FILE = os.path.join(APP_DIR, "world_cup_xgb_model.pkl")

DECAY_HALF_LIFE_DAYS = 365.0
WC_2022_COMPETITION = "FIFA World Cup 2022"

FEATURE_COLS = [
    "home_avg_xG",
    "away_avg_xG",
    "home_avg_xGA",
    "away_avg_xGA",
    "diff_avg_xG",
    "diff_avg_xGA",
    "home_Form",
    "away_Form",
    "diff_Form",
    "elo_diff",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_total_goals_avg",
    "home_avg_GF",
    "away_avg_GF",
    "home_avg_GA",
    "away_avg_GA",
    "home_avg_SoT",
    "away_avg_SoT",
    "is_knockout",
]


def time_decay_weights(dates: pd.Series, *, reference_date=None, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    ref = pd.to_datetime(reference_date) if reference_date is not None else parsed.max()
    age_days = (ref - parsed).dt.days.clip(lower=0).fillna(0)
    return np.exp(-np.log(2.0) * age_days / half_life_days).to_numpy(dtype=float)


def main() -> None:
    print("STEP 3 (World Cup): loading features...")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["Result"].astype(int)
    weights = time_decay_weights(df["Date"])

    print(f"  Training on {len(X):,} matches, {len(feature_cols)} features")

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores: list[float] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights[train_idx])
        preds = model.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], preds)
        cv_scores.append(acc)
        print(f"  CV fold {fold}: accuracy = {acc:.3f}")

    print(f"  Walk-forward CV mean accuracy: {np.mean(cv_scores):.3f}")

    # Tournament holdout: train on all other comps, test on WC 2022 only
    holdout_mask = df["competition"] == WC_2022_COMPETITION if "competition" in df.columns else pd.Series(False, index=df.index)
    if holdout_mask.any() and (~holdout_mask).any():
        train_idx = ~holdout_mask
        test_idx = holdout_mask
        holdout_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
        holdout_model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx.to_numpy()])
        holdout_preds = holdout_model.predict(X[test_idx])
        holdout_proba = holdout_model.predict_proba(X[test_idx])
        holdout_acc = accuracy_score(y[test_idx], holdout_preds)
        holdout_ll = log_loss(y[test_idx], holdout_proba, labels=[0, 1, 2])
        print(f"  WC 2022 holdout ({test_idx.sum()} matches): accuracy = {holdout_acc:.3f}, log-loss = {holdout_ll:.3f}")
        print("  Confusion matrix (rows=true, cols=pred; 0=Away 1=Draw 2=Home):")
        print(confusion_matrix(y[test_idx], holdout_preds, labels=[0, 1, 2]))
        print(classification_report(y[test_idx], holdout_preds, labels=[0, 1, 2], target_names=["Away", "Draw", "Home"]))

    # Final fit on all data
    final_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )
    final_model.fit(X, y, sample_weight=weights)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "classes": [0, 1, 2],
        },
        MODEL_FILE,
    )
    print(f"OK: model saved -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
