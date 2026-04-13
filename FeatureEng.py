import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
import joblib

# =============================================================
# LOAD THE BEST TIME-WEIGHTED ELO FILE
# =============================================================
df = pd.read_csv("premier_league_with_elo_best.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Loaded {len(df):,} matches with best ELO (half_life=1.5)")


# =============================================================
# FEATURE ENGINEERING (rich set like the tennis project)
# =============================================================
def create_features(df):
    df = df.copy()

    # 1. ELO features
    df['elo_diff'] = df['elo_home_before'] - df['elo_away_before']

    # 2. Rolling form (last 5 and last 10 matches)
    for window in [5, 10]:
        # Home team
        df[f'home_win_rate_{window}'] = df.groupby('home_team')['result'].transform(
            lambda x: (x == 'H').rolling(window, min_periods=1).mean())
        df[f'home_draw_rate_{window}'] = df.groupby('home_team')['result'].transform(
            lambda x: (x == 'D').rolling(window, min_periods=1).mean())
        df[f'home_goals_scored_{window}'] = df.groupby('home_team')['home_goals'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())

        # Away team
        df[f'away_win_rate_{window}'] = df.groupby('away_team')['result'].transform(
            lambda x: (x == 'A').rolling(window, min_periods=1).mean())
        df[f'away_draw_rate_{window}'] = df.groupby('away_team')['result'].transform(
            lambda x: (x == 'D').rolling(window, min_periods=1).mean())
        df[f'away_goals_scored_{window}'] = df.groupby('away_team')['away_goals'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())

    # 3. Head-to-head win rate (last 5 meetings)
    def h2h_home(row):
        past = df[(df['date'] < row['date']) &
                  (((df['home_team'] == row['home_team']) & (df['away_team'] == row['away_team'])) |
                   ((df['home_team'] == row['away_team']) & (df['away_team'] == row['home_team'])))].tail(5)
        if len(past) == 0:
            return 0.0
        wins = ((past['home_team'] == row['home_team']) & (past['result'] == 'H')).sum() + \
               ((past['away_team'] == row['home_team']) & (past['result'] == 'A')).sum()
        return wins / len(past)

    df['h2h_home_win_rate'] = df.apply(h2h_home, axis=1)

    # Feature list (you can expand to 81+ later)
    feature_cols = [
        'elo_diff',
        'home_win_rate_5', 'home_win_rate_10',
        'away_win_rate_5', 'away_win_rate_10',
        'home_draw_rate_5', 'away_draw_rate_5',
        'h2h_home_win_rate'
    ]

    return df, feature_cols


df_features, feature_cols = create_features(df)
df_features['target'] = df_features['result'].map({'H': 0, 'D': 1, 'A': 2})

print(f"✅ Features created. Shape: {df_features.shape}")

# =============================================================
# TRAIN XGBoost (proper time-series CV)
# =============================================================
X = df_features[feature_cols]
y = df_features['target']

tscv = TimeSeriesSplit(n_splits=5)
accuracies = []
loglosses = []

print("\nTraining full XGBoost model (5-fold time-series CV)...")
for train_idx, test_idx in tscv.split(X):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    prob = model.predict_proba(X.iloc[test_idx])

    acc = accuracy_score(y.iloc[test_idx], pred)
    ll = log_loss(y.iloc[test_idx], prob)
    accuracies.append(acc)
    loglosses.append(ll)

    print(f"   Fold accuracy: {acc:.4f} | Log-loss: {ll:.4f}")

print(f"\n🎉 FINAL XGBoost Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Mean Log-loss: {np.mean(loglosses):.4f}")

# Save model and features
joblib.dump(model, "xgboost_premier_league_model.pkl")
df_features.to_csv("premier_league_features_with_target.csv", index=False)

print("\n✅ Model and features saved!")
print("You now have a full working model exactly like the tennis project.")