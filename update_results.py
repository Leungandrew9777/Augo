#!/usr/bin/env python3
"""
update_results.py  —  Feed actual results back to improve the model
────────────────────────────────────────────────────────────────────
Usage:
    python update_results.py results.csv

results.csv must have columns:
    date, home_team, away_team, home_goals, away_goals
    (result column H/D/A is derived automatically if absent)

What this does:
    1. Validates and merges new results into premier_league_historical_clean.csv
    2. Recomputes time-weighted ELO  →  premier_league_with_elo_best.csv
    3. Re-engineers features and retrains the XGBoost model
    4. Runs run_pipeline.py to refresh predictions_cache.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

HISTORICAL_FILE = "premier_league_historical_clean.csv"
ELO_FILE        = "premier_league_with_elo_best.csv"
MODEL_FILE      = "xgboost_premier_league_model.pkl"
FEATURES_FILE   = "premier_league_features_with_target.csv"
HALF_LIFE       = 1.5   # best value found in ELO.py tuning

FEATURE_COLS = [
    "elo_diff",
    "home_win_rate_5",  "home_win_rate_10",
    "away_win_rate_5",  "away_win_rate_10",
    "home_draw_rate_5", "away_draw_rate_5",
    "h2h_home_win_rate",
]


# ── Step 1: ingest results ─────────────────────────────────────────────────────

def load_and_validate(path: str, known_teams: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"❌  results.csv is missing columns: {missing}")

    df["date"]       = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)

    if "result" not in df.columns:
        df["result"] = df.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
                      else ("A" if r["home_goals"] < r["away_goals"] else "D"),
            axis=1,
        )

    # Warn on unknown teams but don't block — they'll just get default ELO
    for team in pd.concat([df["home_team"], df["away_team"]]).unique():
        if team not in known_teams:
            print(f"⚠️   '{team}' not in historical data — will be treated as new team")

    return df


# ── Step 2: re-compute ELO ─────────────────────────────────────────────────────

def add_elo_time_weighted(df: pd.DataFrame, half_life=HALF_LIFE, k=32, home_advantage=100) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    df = df.copy()
    df["elo_home_before"] = 1500.0
    df["elo_away_before"] = 1500.0
    df["elo_diff"]        = 0.0
    today = df["date"].max()

    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in ratings: ratings[home] = 1500.0
        if away not in ratings: ratings[away] = 1500.0

        age_years = (today - row["date"]).days / 365.25
        eff_k     = k * np.exp(-age_years / half_life)
        e_home    = 1 / (1 + 10 ** ((ratings[away] - (ratings[home] + home_advantage)) / 400))
        s_home    = {"H": 1.0, "D": 0.5, "A": 0.0}[row["result"]]

        ratings[home] += eff_k * (s_home       - e_home)
        ratings[away] += eff_k * ((1 - s_home) - (1 - e_home))

        df.at[idx, "elo_home_before"] = ratings[home]
        df.at[idx, "elo_away_before"] = ratings[away]
        df.at[idx, "elo_diff"]        = ratings[home] - ratings[away]

    return df


# ── Step 3: feature engineering ───────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["elo_home_before"] - df["elo_away_before"]
    for w in [5, 10]:
        df[f"home_win_rate_{w}"]  = df.groupby("home_team")["result"].transform(
            lambda x: (x == "H").rolling(w, min_periods=1).mean())
        df[f"home_draw_rate_{w}"] = df.groupby("home_team")["result"].transform(
            lambda x: (x == "D").rolling(w, min_periods=1).mean())
        df[f"away_win_rate_{w}"]  = df.groupby("away_team")["result"].transform(
            lambda x: (x == "A").rolling(w, min_periods=1).mean())
        df[f"away_draw_rate_{w}"] = df.groupby("away_team")["result"].transform(
            lambda x: (x == "D").rolling(w, min_periods=1).mean())
    df["h2h_home_win_rate"] = 0.5
    df["target"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    return df


# ── Step 4: retrain ────────────────────────────────────────────────────────────

def retrain(df_features: pd.DataFrame) -> XGBClassifier:
    X = df_features[FEATURE_COLS]
    y = df_features["target"]
    tscv = TimeSeriesSplit(n_splits=5)
    accs: list[float] = []

    for tr, te in tscv.split(X):
        m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          eval_metric="mlogloss")
        m.fit(X.iloc[tr], y.iloc[tr])
        accs.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))

    print(f"   CV accuracy: {np.mean(accs):.4f}  (±{np.std(accs):.4f})")

    final = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          eval_metric="mlogloss")
    final.fit(X, y)
    return final


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python update_results.py <results.csv>")

    results_path = sys.argv[1]
    if not os.path.exists(results_path):
        sys.exit(f"❌  File not found: {results_path}")

    if not os.path.exists(HISTORICAL_FILE):
        sys.exit(f"❌  {HISTORICAL_FILE} not found — this must exist before importing results.")

    # ── 1. Load & merge ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading results from {results_path} …")
    hist = pd.read_csv(HISTORICAL_FILE)
    hist["date"] = pd.to_datetime(hist["date"])
    known_teams  = set(pd.concat([hist["home_team"], hist["away_team"]]).unique())

    new = load_and_validate(results_path, known_teams)
    print(f"      New results: {len(new)} matches")

    # Keep only columns that exist in historical data (forward-compatible)
    shared_cols = [c for c in hist.columns if c in new.columns]
    merged = (
        pd.concat([hist, new[shared_cols]], ignore_index=True)
        .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    added = len(merged) - len(hist)
    print(f"      Added {added} new rows (skipped {len(new) - added} duplicates)")
    merged.to_csv(HISTORICAL_FILE, index=False)
    print(f"✅    Historical data updated → {HISTORICAL_FILE}")

    # ── 2. Recompute ELO ───────────────────────────────────────────────────────
    print(f"\n[2/4] Recomputing time-weighted ELO (half_life={HALF_LIFE}) …")
    df_elo = add_elo_time_weighted(merged)
    df_elo.to_csv(ELO_FILE, index=False)
    print(f"✅    ELO updated → {ELO_FILE}")

    # Print current top 10 as a sanity check
    latest: dict[str, float] = {}
    for _, row in df_elo.iterrows():
        latest[row["home_team"]] = row["elo_home_before"]
        latest[row["away_team"]] = row["elo_away_before"]
    top10 = sorted(latest.items(), key=lambda x: x[1], reverse=True)[:10]
    print("      Current top 10:")
    for rank, (team, elo) in enumerate(top10, 1):
        print(f"        {rank:2}. {team:<28} {elo:.0f}")

    # ── 3. Re-engineer features ────────────────────────────────────────────────
    print(f"\n[3/4] Re-engineering features …")
    df_features = create_features(df_elo)
    df_features.to_csv(FEATURES_FILE, index=False)
    print(f"      Shape: {df_features.shape}")

    # ── 4. Retrain model ───────────────────────────────────────────────────────
    print(f"\n[4/4] Retraining XGBoost model …")
    model = retrain(df_features)
    joblib.dump(model, MODEL_FILE)
    print(f"✅    Model saved → {MODEL_FILE}")

    # ── 5. Refresh predictions cache ──────────────────────────────────────────
    print(f"\n[+]  Refreshing predictions cache …")
    ret = os.system("python run_pipeline.py")
    if ret != 0:
        print("⚠️   run_pipeline.py exited with an error — check fixtures.csv for upcoming matches.")

    print("\n🏁  All done. Restart the app to load the new model and predictions.")


if __name__ == "__main__":
    main()