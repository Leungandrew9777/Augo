import pandas as pd

# =============================================================
# FIXED Part I – Load & Clean Premier League Historical Data
# =============================================================

seasons = ["2526", "2425", "2324", "2223", "2122", "2021", "1920",
           "1819", "1718", "1617", "1516", "1415"]

# Columns we actually need
KEEP_COLUMNS = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}

dfs = []
for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    try:
        # Load full CSV (avoids usecols silently failing on older layouts)
        df_season = pd.read_csv(url, dtype=str)

        # ✅ FIX 2: Drop everything we don't need (Div, Time, all betting cols)
        cols_to_drop = [c for c in df_season.columns if c not in KEEP_COLUMNS]
        df_season = df_season.drop(columns=cols_to_drop, errors='ignore')

        # Verify all required columns survived
        missing = KEEP_COLUMNS - set(df_season.columns)
        if missing:
            print(f"⚠️  Skipped {season} – missing columns: {missing}")
            continue

        # Parse dates (dayfirst because source is DD/MM/YY or DD/MM/YYYY)
        df_season['Date'] = pd.to_datetime(
            df_season['Date'], dayfirst=True, errors='coerce'
        )

        # Convert numeric columns back from str
        for col in ['FTHG', 'FTAG']:
            df_season[col] = pd.to_numeric(df_season[col], errors='coerce')

        dfs.append(df_season)
        print(f"✅ Loaded {season} → {len(df_season):,} matches")

    except Exception as e:
        print(f"⚠️  Skipped {season} → {e}")

# Combine
df = pd.concat(dfs, ignore_index=True)

# Standardize column names
df = df.rename(columns={
    'Date':     'date',
    'HomeTeam': 'home_team',
    'AwayTeam': 'away_team',
    'FTHG':     'home_goals',
    'FTAG':     'away_goals',
    'FTR':      'result'
})

# Team name standardization
team_name_map = {
    "Man City":      "Manchester City",
    "Man Utd":       "Manchester United",
    "Man United":    "Manchester United",
    "Brighton":      "Brighton & Hove Albion",
    "Sheffield W":   "Sheffield Wednesday",
    "Blackburn":     "Blackburn Rovers",
    "Bolton":        "Bolton Wanderers",
    "Derby":         "Derby County",
    "Ipswich":       "Ipswich Town",
    "Leeds":         "Leeds United",
    "Leicester":     "Leicester City",
    "Norwich":       "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "QPR":           "Queens Park Rangers",
    "Tottenham":     "Tottenham Hotspur",
    "West Brom":     "West Bromwich Albion",
    "West Ham":      "West Ham United",
    "Wolves":        "Wolverhampton Wanderers",
}
df['home_team'] = df['home_team'].replace(team_name_map)
df['away_team'] = df['away_team'].replace(team_name_map)

# Sort and drop unparseable dates (critical for ELO)
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['date', 'home_goals', 'away_goals', 'result'])

# ✅ FIX 1: Convert datetime → plain date string before saving
#    Without this, Excel sees a timestamp and renders ######
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df.to_csv("premier_league_historical_clean.csv", index=False)

print("\n🎉 SUCCESS! Clean dataset ready:")
print(f"   Total matches : {len(df):,}")
print(f"   Date range    : {df['date'].min()} to {df['date'].max()}")
print(f"   Columns       : {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']].head())