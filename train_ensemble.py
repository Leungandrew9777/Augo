# train_ensemble.py
import pandas as pd
import numpy as np
import inspect
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_poisson_deviance
import joblib

from model_artifacts import WeightedSoftVotingClassifier

print("STEP 5: Loading rich features for ensemble training...")
df = pd.read_csv("premier_league_with_elo_best.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

DECAY_HALF_LIFE_DAYS = 365.0


def time_decay_weights(dates: pd.Series, *, reference_date=None, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    ref = pd.to_datetime(reference_date) if reference_date is not None else parsed.max()
    age_days = (ref - parsed).dt.days.clip(lower=0).fillna(0)
    return np.exp(-np.log(2.0) * age_days / half_life_days).to_numpy(dtype=float)


def fit_lr_pipeline(estimator: Pipeline, X_fit: pd.DataFrame, y_fit, sample_weight):
    estimator.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    return estimator


def fit_weighted_classifier(X_fit: pd.DataFrame, y_fit: pd.Series, sample_weight) -> WeightedSoftVotingClassifier:
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**lr_kwargs)),
        ]
    )
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42)
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )

    fitted = [
        fit_lr_pipeline(clone(lr), X_fit, y_fit, sample_weight),
        clone(rf).fit(X_fit, y_fit, sample_weight=sample_weight),
        clone(xgb).fit(X_fit, y_fit, sample_weight=sample_weight),
    ]
    return WeightedSoftVotingClassifier(
        estimators_=fitted,
        weights=[1.0, 1.0, 2.0],
        classes_=np.array([0, 1, 2]),
        feature_names_in_=np.array(list(X_fit.columns), dtype=object),
    )


def fit_goal_model(X_fit: pd.DataFrame, y_fit: pd.Series, sample_weight):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
        ]
    )
    model.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    return model

# Football-only feature set. Bookmaker implied probabilities are intentionally
# excluded so the model remains independent from the market.
feature_cols = [
    "elo_diff",
    "home_Form", "away_Form",                    # avg points (form)
    "diff_Form",
    "home_avg_GF", "away_avg_GF",
    "home_avg_GA", "away_avg_GA",
    "home_avg_SoT", "away_avg_SoT",
    "home_avg_SoTAgainst", "away_avg_SoTAgainst",
    "home_avg_Shots", "away_avg_Shots",
    "home_avg_ShotsAgainst", "away_avg_ShotsAgainst",
    "home_avg_xG", "away_avg_xG",
    "home_avg_xGA", "away_avg_xGA",
    "home_avg_xG_overperf", "away_avg_xG_overperf",
    "diff_avg_GF", "diff_avg_GA", "diff_avg_SoT", "diff_avg_Shots",
    "diff_avg_SoTAgainst", "diff_avg_ShotsAgainst",
    "diff_avg_xG", "diff_avg_xGA", "diff_avg_xG_overperf",
    "h2h_home_wins", "h2h_draws", "h2h_total_goals_avg",
]

# Use only columns that actually exist in the file
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df["Result"].astype(int)  # 0=Away, 1=Draw, 2=Home
y_home_goals = df["FTHG"].astype(float)
y_away_goals = df["FTAG"].astype(float)

print(f"Training ensemble on {len(X):,} matches with {len(feature_cols)} features")

# `multi_class` is deprecated from sklearn 1.5; omit it and use defaults (multiclass + lbfgs).
lr_kwargs = {"max_iter": 3000, "C": 0.5, "solver": "lbfgs"}
_sk_parts = sklearn.__version__.split(".")
_sk_major, _sk_minor = int(_sk_parts[0]), int(_sk_parts[1]) if len(_sk_parts) > 1 else 0
if (_sk_major, _sk_minor) < (1, 5) and "multi_class" in inspect.signature(LogisticRegression).parameters:
    lr_kwargs["multi_class"] = "multinomial"

# STEP 6: Walk-forward backtest (correct for sports data)
tscv = TimeSeriesSplit(n_splits=5)
scores = []
home_devs = []
away_devs = []
for train_idx, test_idx in tscv.split(X):
    fold_ref = df.iloc[train_idx]["Date"].max()
    weights = time_decay_weights(df.iloc[train_idx]["Date"], reference_date=fold_ref)
    ensemble = fit_weighted_classifier(X.iloc[train_idx], y.iloc[train_idx], weights)
    home_goal_model = fit_goal_model(X.iloc[train_idx], y_home_goals.iloc[train_idx], weights)
    away_goal_model = fit_goal_model(X.iloc[train_idx], y_away_goals.iloc[train_idx], weights)

    pred = ensemble.predict(X.iloc[test_idx])
    scores.append(accuracy_score(y.iloc[test_idx], pred))
    pred_home_goals = np.clip(home_goal_model.predict(X.iloc[test_idx]), 0.05, None)
    pred_away_goals = np.clip(away_goal_model.predict(X.iloc[test_idx]), 0.05, None)
    home_devs.append(mean_poisson_deviance(y_home_goals.iloc[test_idx], pred_home_goals))
    away_devs.append(mean_poisson_deviance(y_away_goals.iloc[test_idx], pred_away_goals))

print(f"[OK] STEP 6 Walk-forward CV Accuracy: {np.mean(scores):.4f}")
print(f"[OK] Goal model mean Poisson deviance: home={np.mean(home_devs):.4f}, away={np.mean(away_devs):.4f}")

# STEP 7: Final model on all data + save exact filenames your app expects
final_weights = time_decay_weights(df["Date"])
ensemble = fit_weighted_classifier(X, y, final_weights)
goal_model_home = fit_goal_model(X, y_home_goals, final_weights)
goal_model_away = fit_goal_model(X, y_away_goals, final_weights)
joblib.dump(ensemble, "xgboost_premier_league_model.pkl")
joblib.dump(goal_model_home, "goal_model_home.pkl")
joblib.dump(goal_model_away, "goal_model_away.pkl")
df.to_csv("premier_league_with_elo_best.csv", index=False)

print("\nSTEP 5 + 6 + 7 COMPLETE")
print("   Model saved -> xgboost_premier_league_model.pkl")
print("   Goal model home saved -> goal_model_home.pkl")
print("   Goal model away saved -> goal_model_away.pkl")
print("   Data saved   -> premier_league_with_elo_best.csv")
print("   Your existing app.py and run_pipeline.py will now use the new ensemble model.")