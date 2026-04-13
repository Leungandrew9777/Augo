#!/usr/bin/env python3
"""
fetch_understat_xg.py — Pull Premier League xG data using understatapi
Merges xG columns INTO premier_league_historical_clean.csv without
overwriting existing columns (home_goals, away_goals, result, etc.)
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

TEAM_NAME_MAP = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Tottenham Hotspur": "Tottenham Hotspur",       
    "Tottenham": "Tottenham Hotspur",
    "West Ham United": "West Ham United",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds United": "Leeds United",
    "Leeds": "Leeds United",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Wolves": "Wolverhampton Wanderers",
}
# ========================================================

HISTORICAL_FILE = "premier_league_historical_clean.csv"

def normalize_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)

print("🚀 Starting Understat xG fetch using understatapi...")

all_matches = []
with UnderstatClient() as understat:
    for season in tqdm(range(START_SEASON, END_SEASON + 1), desc="Seasons"):
        try:
            match_data = understat.league(league=LEAGUE).get_match_data(season=str(season))
            
            for m in match_data:
                if not m.get("isResult", False):
                    continue

                date_str = m.get("datetime") or m.get("date") or ""

                h = m.get("h") or {}
                a = m.get("a") or {}

                home_team = normalize_team(
                    m.get("h_team") or h.get("title") or h.get("name") or h.get("team") or ""
                )
                away_team = normalize_team(
                    m.get("a_team") or a.get("title") or a.get("name") or a.get("team") or ""
                )
                
                if not home_team or not away_team:
                    continue

                xg = m.get("xG") or {}
                try:
                    home_xg = float(xg.get("h", h.get("xG", 0.0)) or 0.0)
                    away_xg = float(xg.get("a", a.get("xG", 0.0)) or 0.0)
                except Exception:
                    home_xg = away_xg = 0.0

                all_matches.append({
                    "date":      pd.to_datetime(date_str, errors="coerce").date(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_xg":   round(home_xg, 3),
                    "away_xg":   round(away_xg, 3),
                })
                
        except Exception as e:
            print(f"⚠️  Season {season} skipped: {e}")

# ── Build xG dataframe ────────────────────────────────────────────────────────
df_xg = pd.DataFrame(all_matches)
if df_xg.empty:
    raise SystemExit("No matches returned from Understat — API may have failed.")

df_xg["date"] = pd.to_datetime(df_xg["date"], errors="coerce")
df_xg = df_xg.dropna(subset=["date", "home_team", "away_team"])
df_xg = df_xg.drop_duplicates(subset=["date", "home_team", "away_team"])

print(f"\n✅ Fetched {len(df_xg):,} xG records from Understat.")

# ── Merge INTO existing historical file ───────────────────────────────────────
import os
if not os.path.exists(HISTORICAL_FILE):
    raise SystemExit(f"❌ {HISTORICAL_FILE} not found — run your historical data pipeline first.")

existing = pd.read_csv(HISTORICAL_FILE)
existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
print(f"   Existing historical rows: {len(existing):,}")

# Drop stale xg columns if they exist from a previous run
for col in ["home_xg", "away_xg"]:
    if col in existing.columns:
        existing = existing.drop(columns=[col])

# Left join: keeps ALL existing rows; unmatched ones get NaN for xg
merged = existing.merge(
    df_xg[["date", "home_team", "away_team", "home_xg", "away_xg"]],
    on=["date", "home_team", "away_team"],
    how="left",
)

matched   = merged["home_xg"].notna().sum()
unmatched = merged["home_xg"].isna().sum()
print(f"   Matched xG:   {matched:,} rows")
print(f"   Unmatched:    {unmatched:,} rows (xg will be NaN — check team name mapping)")

merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
merged.to_csv(HISTORICAL_FILE, index=False)
print(f"\n✅ Saved → {HISTORICAL_FILE}  (original columns preserved, xg added)")
print(f"   Date range: {merged['date'].min()} to {merged['date'].max()}")