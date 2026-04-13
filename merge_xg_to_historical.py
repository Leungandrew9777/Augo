import pandas as pd

hist = pd.read_csv("premier_league_historical_clean.csv")
hist["date"] = pd.to_datetime(hist["date"]).dt.date

xg = pd.read_csv("premier_league_xg.csv")
xg["date"] = pd.to_datetime(xg["date"]).dt.date

merged = hist.merge(
    xg[["date", "home_team", "away_team", "home_xg", "away_xg"]],
    on=["date", "home_team", "away_team"],
    how="left"
)

print(f"✅ Added xG to {merged['home_xg'].notna().sum():,} matches")
merged.to_csv("premier_league_historical_clean.csv", index=False)
print("   File updated — ready for ELO tuning!")