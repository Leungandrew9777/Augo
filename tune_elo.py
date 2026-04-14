#!/usr/bin/env python3
"""
tune_elo.py — Grid search for ELO + xG hyperparameters
Now includes xg_blend_weight and xg_margin_weight
"""
import itertools, time, os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

# ====================== CONFIG ======================
HISTORICAL_FILE = "premier_league_historical_clean.csv"
ELO_OUT         = "premier_league_with_elo_best.csv"
MODEL_OUT       = "xgboost_premier_league_model.pkl"
FEATURES_OUT    = "premier_league_features_with_target.csv"

# Expanded grid (still runs in ~10-20 min on a laptop)
HALF_LIFE_GRID   = [1.0, 1.5, 2.5, 3.0]
K_GRID           = [32]
HOME_ADV_GRID    = [110]
XG_BLEND_GRID    = [0.15, 1.0]   # how much xG share influences the result score
XG_MARGIN_GRID   = [0.1, 0.75]           # extra K scaling per xG goal margin

N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 6
# ====================================================

def add_elo_time_weighted(df: pd.DataFrame, half_life: float, k: int, home_advantage: int,
                          xg_blend_weight: float = 0.0, xg_margin_weight: float = 0.0) -> pd.DataFrame:
    """Time-weighted ELO with optional xG blending + margin-adjusted K."""
    ratings: dict[str, float] = {}
    df = df.copy()
    df["elo_home_before"] = 1500.0
    df["elo_away_before"] = 1500.0
    df["elo_diff"] = 0.0
    today = df["date"].max()

    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in ratings: ratings[home] = 1500.0
        if away not in ratings: ratings[away] = 1500.0

        age_years = (today - row["date"]).days / 365.25
        eff_k = k * np.exp(-age_years / half_life)

        # --- xG MARGIN SCALING OF K (optional) ---
        if xg_margin_weight > 0 and "home_xg" in row and "away_xg" in row and not pd.isna(row["home_xg"]):
            xg_margin = abs(row["home_xg"] - row["away_xg"])
            xg_factor = 1 + (xg_margin * xg_margin_weight)
            eff_k *= xg_factor

        # --- xG BLENDING OF ACTUAL RESULT (primary method) ---
        if row["result"] == "H":
            s_result_home, s_result_away = 1.0, 0.0
        elif row["result"] == "D":
            s_result_home, s_result_away = 0.5, 0.5
        else:
            s_result_home, s_result_away = 0.0, 1.0

        s_home = s_result_home
        s_away = s_result_away

        if xg_blend_weight > 0 and "home_xg" in row and "away_xg" in row and not pd.isna(row["home_xg"]):
            total_xg = row["home_xg"] + row["away_xg"] + 1e-9
            xg_home_share = row["home_xg"] / total_xg
            s_home = (1 - xg_blend_weight) * s_result_home + xg_blend_weight * xg_home_share
            s_away = 1 - s_home

        # --- Standard ELO update with (possibly blended) scores ---
        r_home = ratings[home] + home_advantage
        r_away = ratings[away]
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))

        ratings[home] += eff_k * (s_home - e_home)
        ratings[away] += eff_k * (s_away - (1 - e_home))

        df.at[idx, "elo_home_before"] = ratings[home]
        df.at[idx, "elo_away_before"] = ratings[away]
        df.at[idx, "elo_diff"]        = ratings[home] - ratings[away]

    return df


def create_features(df_elo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Same feature engineering as before (unchanged)."""
    df = df_elo.copy()
    df["elo_diff"] = df["elo_home_before"] - df["elo_away_before"]

    for window in [5, 10]:
        df[f"home_win_rate_{window}"] = df.groupby("home_team")["result"].transform(
            lambda x: (x == "H").rolling(window, min_periods=1).mean())
        df[f"home_draw_rate_{window}"] = df.groupby("home_team")["result"].transform(
            lambda x: (x == "D").rolling(window, min_periods=1).mean())
        df[f"away_win_rate_{window}"] = df.groupby("away_team")["result"].transform(
            lambda x: (x == "A").rolling(window, min_periods=1).mean())
        df[f"away_draw_rate_{window}"] = df.groupby("away_team")["result"].transform(
            lambda x: (x == "D").rolling(window, min_periods=1).mean())

    df["h2h_home_win_rate"] = 0.5
    df["target"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    feature_cols = ["elo_diff", "home_win_rate_5", "home_win_rate_10",
                    "away_win_rate_5", "away_win_rate_10",
                    "home_draw_rate_5", "away_draw_rate_5", "h2h_home_win_rate"]
    return df, feature_cols


def evaluate_params(df_hist: pd.DataFrame, half_life: float, k: int, home_adv: int,
                    blend_w: float, margin_w: float) -> float:
    start = time.time()
    df_elo = add_elo_time_weighted(df_hist, half_life, k, home_adv, blend_w, margin_w)
    df_feat, feature_cols = create_features(df_elo)

    X = df_feat[feature_cols]
    y = df_feat["target"]
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for tr, te in tscv.split(X):
        model = XGBClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                              max_depth=MAX_DEPTH, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, eval_metric="mlogloss")
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        scores.append(accuracy_score(y.iloc[te], pred))

    mean_acc = float(np.mean(scores))
    elapsed = time.time() - start
    print(f"  hl={half_life:4.1f} K={k:2d} ha={home_adv:3d} blend={blend_w:.2f} margin={margin_w:.2f} "
          f"→ CV acc={mean_acc:.4f} ({elapsed:.1f}s)")
    return mean_acc


def main():
    if not os.path.exists(HISTORICAL_FILE):
        print(f"❌ {HISTORICAL_FILE} not found.")
        return

    df_hist = pd.read_csv(HISTORICAL_FILE)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df_hist):,} matches with xG columns: {'home_xg' in df_hist.columns}\n")

    print("Starting full ELO + xG grid search...")
    best_acc = -1
    best_params = None
    best_elo_df = None

    grid = list(itertools.product(HALF_LIFE_GRID, K_GRID, HOME_ADV_GRID, XG_BLEND_GRID, XG_MARGIN_GRID))
    total = len(grid)

    for i, (hl, k, ha, blend, margin) in enumerate(grid, 1):
        print(f"[{i:3d}/{total}] Testing parameters...")
        acc = evaluate_params(df_hist, hl, k, ha, blend, margin)

        if acc > best_acc:
            best_acc = acc
            best_params = {"half_life": hl, "k": k, "home_advantage": ha,
                           "xg_blend_weight": blend, "xg_margin_weight": margin}
            best_elo_df = add_elo_time_weighted(df_hist, hl, k, ha, blend, margin)

    # === RESULTS ===
    print("\n" + "="*70)
    print("🏆 BEST ELO + xG PARAMETERS")
    for k, v in best_params.items():
        print(f"   {k:18} = {v}")
    print(f"→ Best CV Accuracy = {best_acc:.4f}")
    print("="*70)

    best_elo_df.to_csv(ELO_OUT, index=False)
    print(f"✅ Best ELO saved → {ELO_OUT}")

    df_final, feat_cols = create_features(best_elo_df)
    df_final.to_csv(FEATURES_OUT, index=False)

    final_model = XGBClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                                max_depth=MAX_DEPTH, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, eval_metric="mlogloss")
    final_model.fit(df_final[feat_cols], df_final["target"])
    joblib.dump(final_model, MODEL_OUT)
    print(f"✅ Final model saved → {MODEL_OUT}")

    print("\nRefreshing predictions...")
    os.system("python run_pipeline.py")
    print("\n🎉 Done! Restart Reflex with `reflex run` to use the xG-enhanced model.")


if __name__ == "__main__":
    main()