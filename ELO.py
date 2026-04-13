import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
# Load your full clean data
df = pd.read_csv("premier_league_historical_clean.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Loaded {len(df):,} matches for tuning.\n")
# ==================== TIME-WEIGHTED ELO ====================
def add_elo_ratings_time_weighted(df, half_life=2.0, k=32, home_advantage=100):
    ratings = {}
    df = df.copy()
    df['elo_home_before'] = 1500.0
    df['elo_away_before'] = 1500.0
    df['elo_diff'] = 0.0
    today = df['date'].max()
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        if home not in ratings: ratings[home] = 1500.0
        if away not in ratings: ratings[away] = 1500.0
        age_years = (today - row['date']).days / 365.25
        weight = np.exp(-age_years / half_life)
        effective_k = k * weight
        r_home = ratings[home] + home_advantage
        r_away = ratings[away]
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        if row['result'] == 'H':
            s_home, s_away = 1.0, 0.0
        elif row['result'] == 'D':
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0
        ratings[home] = ratings[home] + effective_k * (s_home - e_home)
        ratings[away] = ratings[away] + effective_k * (s_away - (1 - e_home))
        df.at[idx, 'elo_home_before'] = ratings[home]
        df.at[idx, 'elo_away_before'] = ratings[away]
        df.at[idx, 'elo_diff'] = ratings[home] - ratings[away]
    return df
# ==================== TUNING LOOP ====================
half_life_values = [1.5, 1.8, 2.0, 2.2, 2.5]
best_acc = 0
best_hl = None
best_top20 = None
for hl in half_life_values:
    print(f"Testing half_life = {hl} ...")
    df_elo = add_elo_ratings_time_weighted(df, half_life=hl)
    # Current top 20
    latest = {}
    for _, row in df_elo.iterrows():
        latest[row['home_team']] = row['elo_home_before']
        latest[row['away_team']] = row['elo_away_before']
    top20 = pd.Series(latest).sort_values(ascending=False).head(20)
    # Simple features for fast CV
    X = df_elo[['elo_diff']]
    y = df_elo['result'].map({'H':0, 'D':1, 'A':2})
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in tscv.split(X):
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        scores.append(accuracy_score(y.iloc[te], pred))
    mean_acc = np.mean(scores)
    print(f"   → Accuracy: {mean_acc:.4f} | Top team: {top20.index[0]} ({top20.iloc[0]:.0f})")
    if "Hull" not in top20.index[:15] and "Stoke" not in top20.index[:15]:
        print("   ✓ Hull & Stoke OUT of top 15")
    else:
        print("   ⚠️  Hull or Stoke still high")
    if mean_acc > best_acc:
        best_acc = mean_acc
        best_hl = hl
        best_top20 = top20
print(f"\n🏆 BEST half_life = {best_hl} with accuracy {best_acc:.4f}")
print("\nFinal Top 20 with best half_life:")
print(best_top20)
# Save best version
best_df = add_elo_ratings_time_weighted(df, half_life=best_hl)
best_df.to_csv("premier_league_with_elo_best.csv", index=False)
print("\nBest ELO file saved as 'premier_league_with_elo_best.csv'")