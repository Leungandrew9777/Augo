#!/usr/bin/env python3
"""
fetch_understat_xg.py — Pull Premier League xG data using understatapi
Outputs: premier_league_xg.csv (ready to merge with your historical data)
"""

import pandas as pd
from understatapi import UnderstatClient
from tqdm import tqdm
import time
from datetime import datetime

# ========================= CONFIG =========================
LEAGUE = "EPL"
START_SEASON = 2014   # 2014/15 season
END_SEASON = 2026     # up to current / future season (Understat updates live)

# Your exact team names from the app (for perfect matching)
TEAM_NAME_MAP = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "West Ham United": "West Ham United",
    "Newcastle": "Newcastle",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds United": "Leeds United",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    # Add any missing teams here if needed
}
# ========================================================

def normalize_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)

print("🚀 Starting Understat xG fetch using understatapi...")

all_matches = []
with UnderstatClient() as understat:
    for season in tqdm(range(START_SEASON, END_SEASON + 1), desc="Seasons"):
        try:
            # Get all matches for the season via one team (covers the entire league)
            match_data = understat.team(team="Manchester_United").get_match_data(season=str(season))
            
            for m in match_data:
                match_id = m["id"]
                date_str = m["date"]  # Understat format: YYYY-MM-DD HH:MM
                home_team = normalize_team(m.get("h_team", ""))
                away_team = normalize_team(m.get("a_team", ""))
                
                if not home_team or not away_team:
                    continue

                # Get shot data to calculate xG
                try:
                    shots = understat.match(match=match_id).get_shot_data()
                    home_xg = sum(float(s["xG"]) for s in shots if s["h_a"] == "h")
                    away_xg = sum(float(s["xG"]) for s in shots if s["h_a"] == "a")
                except Exception:
                    home_xg = away_xg = 0.0  # fallback

                all_matches.append({
                    "date": pd.to_datetime(date_str).date(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_xg": round(home_xg, 3),
                    "away_xg": round(away_xg, 3),
                })
                
                time.sleep(0.3)  # polite delay to avoid rate-limiting
                
        except Exception as e:
            print(f"⚠️  Season {season} skipped: {e}")

# Save results
df_xg = pd.DataFrame(all_matches)
if df_xg.empty:
    raise SystemExit(
        "No matches were returned from Understat. "
        "This usually means the API call failed or returned an unexpected schema."
    )

# Defensive normalization in case upstream keys ever change.
df_xg.columns = [str(c).strip().lower() for c in df_xg.columns]
if "date" not in df_xg.columns:
    raise KeyError(f"'date' column missing. Columns returned: {list(df_xg.columns)}")

# Ensure sortable datetime (keep output as YYYY-MM-DD in CSV).
df_xg["date"] = pd.to_datetime(df_xg["date"], errors="coerce").dt.date
df_xg = df_xg.dropna(subset=["date", "home_team", "away_team"])
df_xg = df_xg.drop_duplicates(subset=["date", "home_team", "away_team"])
df_xg = df_xg.sort_values("date").reset_index(drop=True)

df_xg.to_csv("premier_league_xg.csv", index=False)
print(f"\n✅ Success! Saved {len(df_xg):,} matches with xG → premier_league_xg.csv")
print(f"   Date range: {df_xg['date'].min()} to {df_xg['date'].max()}")
print("\nNext step: Run the merge script below to add xG to your historical file.")