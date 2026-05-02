# data_pipeline.py
import pandas as pd
from pathlib import Path
import numpy as np

class FootballDataLoader:
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    LEAGUES = {"E0": "Premier League"}

    # Core columns we actually need for the model + odds
    COLUMNS_TO_KEEP = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",           # Full-time result & goals
        "HTHG", "HTAG", "HTR",           # Half-time (optional but useful)
        "HS", "AS", "HST", "AST",        # Shots & shots on target
        "HF", "AF", "HC", "AC",          # Fouls & corners
        "HY", "AY", "HR", "AR",          # Cards
        "B365H", "B365D", "B365A"        # Bet365 odds (key feature)
    ]

    def __init__(self, seasons: list[str]):
        self.seasons = seasons   # e.g. ["2526", "2425", "2324", ...]

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Load one season/league CSV with robust parsing."""
        url = f"{self.BASE_URL}/{season}/{league}.csv"
        try:
            # This combination works for all seasons including 2526
            df = pd.read_csv(
                url,
                encoding="ISO-8859-1",      # ← This fixes most parsing issues
                on_bad_lines="skip",
                low_memory=False
            )
            # Keep only columns that actually exist
            available_cols = [c for c in self.COLUMNS_TO_KEEP if c in df.columns]
            df = df[available_cols].copy()

            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df["League"] = self.LEAGUES.get(league, league)
            df["Season"] = season
            print(f"✓ {self.LEAGUES.get(league)} {season}: {len(df)} matches")
            return df
        except Exception as e:
            print(f"⚠️  Failed to load {league}/{season}: {e}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        frames = []
        for league in self.LEAGUES:
            for season in self.seasons:
                df = self.load_season(league, season)
                if not df.empty:
                    frames.append(df)
        result = pd.concat(frames, ignore_index=True)
        print(f"\n✅ Total matches loaded: {len(result):,}")
        return result


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        numeric_cols = [
            "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
            "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode result for model: H=2, D=1, A=0
        result_map = {"H": 2, "D": 1, "A": 0}
        df["Result"] = df["FTR"].map(result_map)
        df = df.dropna(subset=["Result"])
        df["Result"] = df["Result"].astype(int)
        return df      


# ====================== RUN THIS ======================
if __name__ == "__main__":
    # Include the current season (2526) + all previous you want
    loader = FootballDataLoader(
        seasons=["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617", "1516", "1415", "1314"]
    )
    raw_data = loader.load_all()

    clean_data = DataCleaner.clean(raw_data)
    clean_data.to_csv("premier_league_historical_clean.csv", index=False)

    print(f"✅ Saved {len(clean_data):,} cleaned matches → premier_league_historical_clean.csv")
    print("Now run the feature engineering + training steps below.")
