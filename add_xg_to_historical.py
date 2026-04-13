import pandas as pd

# Load your current clean historical file
hist = pd.read_csv("premier_league_historical_clean.csv")
hist["date"] = pd.to_datetime(hist["date"])

# Load your xG source file (adjust path/columns as needed)
xg_df = pd.read_csv("your_xg_source.csv")          # ← change filename
xg_df["date"] = pd.to_datetime(xg_df["date"])

# Merge on date + teams (exact match)
merged = hist.merge(
    xg_df[["date", "home_team", "away_team", "home_xg", "away_xg"]],
    on=["date", "home_team", "away_team"],
    how="left"
)

print(f"Added xG to {merged['home_xg'].notna().sum():,} matches")
merged.to_csv("premier_league_historical_clean.csv", index=False)
print("✅ Historical file updated with xG columns")