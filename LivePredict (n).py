import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =============================================================
# 1. LOAD MODEL + BEST ELO FILE
# =============================================================
model = joblib.load("xgboost_premier_league_model.pkl")
df_elo = pd.read_csv("premier_league_with_elo_best.csv")
df_elo['date'] = pd.to_datetime(df_elo['date'])

print("✅ Model and ELO loaded successfully")

# =============================================================
# 2. FULL MATCHWEEK FIXTURES (10 matches – April 11-12 2026)
# =============================================================
upcoming = pd.DataFrame([
    {"date": "2026-04-11", "home_team": "West Ham United",              "away_team": "Wolverhampton Wanderers"},
    {"date": "2026-04-11", "home_team": "Arsenal",     "away_team": "Bournemouth"},
    {"date": "2026-04-11", "home_team": "Brentford",             "away_team": "Everton"},
    {"date": "2026-04-11", "home_team": "Burnley",   "away_team": "Brighton & Hove Albion"},
    {"date": "2026-04-12", "home_team": "Liverpool",   "away_team": "Fulham"},
    {"date": "2026-04-12", "home_team": "Crystal Palace",     "away_team": "Newcastle"},
    {"date": "2026-04-12", "home_team": "Nottingham Forest",   "away_team": "Aston Villa"},
    {"date": "2026-04-12", "home_team": "Sunderland",      "away_team": "Tottenham Hotspur"},
    {"date": "2026-04-12", "home_team": "Chelsea",         "away_team": "Manchester City"},
    {"date": "2026-04-14", "home_team": "Manchester United", "away_team": "Leeds United"}
])

upcoming['date'] = pd.to_datetime(upcoming['date'])
print(f"\nPredicting full matchweek – {len(upcoming)} fixtures...")


# =============================================================
# 3. VALIDATION (prevents hallucinated / misspelled teams)
# =============================================================
def validate_fixtures(upcoming, df_elo):
    valid_teams = set(pd.concat([df_elo['home_team'], df_elo['away_team']]).unique())
    errors = []
    for idx, row in upcoming.iterrows():
        home = row['home_team']
        away = row['away_team']
        if home not in valid_teams:
            suggestion = difflib.get_close_matches(home, valid_teams, n=1, cutoff=0.6)
            msg = f"❌ Invalid home team: '{home}'"
            if suggestion: msg += f" → Did you mean '{suggestion[0]}'?"
            errors.append(msg)
        if away not in valid_teams:
            suggestion = difflib.get_close_matches(away, valid_teams, n=1, cutoff=0.6)
            msg = f"❌ Invalid away team: '{away}'"
            if suggestion: msg += f" → Did you mean '{suggestion[0]}'?"
            errors.append(msg)
    if errors:
        print("\n" + "=" * 80)
        print("🚫 FIXTURE VALIDATION FAILED")
        print("=" * 80)
        for e in errors: print(e)
        print("\nValid teams:", ", ".join(sorted(valid_teams)))
        return False
    print("✅ All 10 fixtures validated — team names are correct.")
    return True


import difflib  # (add this import at the very top of the file if not already there)

if not validate_fixtures(upcoming, df_elo):
    raise SystemExit("Invalid fixtures — please correct team names.")


# =============================================================
# 4. COMPUTE CURRENT ELO + BUILD FEATURES + PREDICT
# =============================================================
def compute_current_elo(upcoming, df_elo):
    """Uses the MOST RECENT ELO for each team"""
    latest_elo = {}
    for team in pd.concat([df_elo['home_team'], df_elo['away_team']]).unique():
        team_matches = df_elo[(df_elo['home_team'] == team) | (df_elo['away_team'] == team)].copy()
        if len(team_matches) > 0:
            last_row = team_matches.sort_values('date').iloc[-1]
            latest_elo[team] = last_row['elo_home_before'] if last_row['home_team'] == team else last_row[
                'elo_away_before']
        else:
            latest_elo[team] = 1500.0
    upcoming['elo_home'] = upcoming['home_team'].map(latest_elo)
    upcoming['elo_away'] = upcoming['away_team'].map(latest_elo)
    upcoming['elo_diff'] = upcoming['elo_home'] - upcoming['elo_away']
    return upcoming


def predict_upcoming(upcoming, df_elo):
    """Builds ALL 8 features the XGBoost model was trained on"""
    upcoming = compute_current_elo(upcoming, df_elo)

    for window in [5, 10]:
        upcoming[f'home_win_rate_{window}'] = upcoming['home_team'].apply(
            lambda t: df_elo[df_elo['home_team'] == t].tail(window)['result'].eq('H').mean()
            if len(df_elo[df_elo['home_team'] == t]) > 0 else 0.5)
        upcoming[f'home_draw_rate_{window}'] = upcoming['home_team'].apply(
            lambda t: df_elo[df_elo['home_team'] == t].tail(window)['result'].eq('D').mean()
            if len(df_elo[df_elo['home_team'] == t]) > 0 else 0.3)
        upcoming[f'away_win_rate_{window}'] = upcoming['away_team'].apply(
            lambda t: df_elo[df_elo['away_team'] == t].tail(window)['result'].eq('A').mean()
            if len(df_elo[df_elo['away_team'] == t]) > 0 else 0.5)
        upcoming[f'away_draw_rate_{window}'] = upcoming['away_team'].apply(
            lambda t: df_elo[df_elo['away_team'] == t].tail(window)['result'].eq('D').mean()
            if len(df_elo[df_elo['away_team'] == t]) > 0 else 0.3)

    upcoming['h2h_home_win_rate'] = 0.5

    feature_cols = [
        'elo_diff',
        'home_win_rate_5', 'home_win_rate_10',
        'away_win_rate_5', 'away_win_rate_10',
        'home_draw_rate_5', 'away_draw_rate_5',
        'h2h_home_win_rate'
    ]

    X_pred = upcoming[feature_cols]
    probs = model.predict_proba(X_pred)

    upcoming['prob_home'] = probs[:, 0]
    upcoming['prob_draw'] = probs[:, 1]
    upcoming['prob_away'] = probs[:, 2]

    upcoming['fair_odds_home'] = 1 / upcoming['prob_home']
    upcoming['fair_odds_draw'] = 1 / upcoming['prob_draw']
    upcoming['fair_odds_away'] = 1 / upcoming['prob_away']

    return upcoming


# Run prediction
predictions = predict_upcoming(upcoming, df_elo)

# =============================================================
# 5. DISPLAY RESULTS
# =============================================================
print("\n" + "=" * 100)
print("🚀 FULL PREMIER LEAGUE MATCHWEEK PREDICTIONS (10 fixtures)")
print("=" * 100)

for _, row in predictions.iterrows():
    print(f"\n{row['date'].date()} | {row['home_team']} vs {row['away_team']}")
    print(f"Model probs : Home {row['prob_home']:.1%} | Draw {row['prob_draw']:.1%} | Away {row['prob_away']:.1%}")
    print(
        f"Fair odds   : Home {row['fair_odds_home']:.2f} | Draw {row['fair_odds_draw']:.2f} | Away {row['fair_odds_away']:.2f}")

predictions.to_csv("matchweek_predictions.csv", index=False)
print("\n✅ Full matchweek saved to 'matchweek_predictions.csv'")