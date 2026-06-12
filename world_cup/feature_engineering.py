#!/usr/bin/env python3
"""
Step 2: rolling international features (10-game xG/form), ELO, H2H -> international_with_features.csv
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

_parent_fe_path = os.path.join(ROOT_DIR, "feature_engineering.py")
_spec = importlib.util.spec_from_file_location("augo_feature_engineering", _parent_fe_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load Augo feature_engineering from {_parent_fe_path}")
_augo_fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_augo_fe)
FootballELO = _augo_fe.FootballELO
add_h2h_features = _augo_fe.add_h2h_features

INPUT_FILE = os.path.join(APP_DIR, "international_matches_clean.csv")
OUTPUT_FILE = os.path.join(APP_DIR, "international_with_features.csv")

ROLLING_WINDOW = 10
MIN_PERIODS = 1

FEATURE_COLS = [
    "home_avg_xG",
    "away_avg_xG",
    "home_avg_xGA",
    "away_avg_xGA",
    "diff_avg_xG",
    "diff_avg_xGA",
    "home_Form",
    "away_Form",
    "diff_Form",
    "elo_diff",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_total_goals_avg",
    "home_avg_GF",
    "away_avg_GF",
    "home_avg_GA",
    "away_avg_GA",
    "home_avg_SoT",
    "away_avg_SoT",
    "is_knockout",
]


class InternationalFeatureEngineer:
    """Rolling stats pooled across all prior internationals per team (not home/away split)."""

    def __init__(self, window: int = ROLLING_WINDOW, min_periods: int = MIN_PERIODS):
        self.window = window
        self.min_periods = min_periods

    def _team_match_records(self, df: pd.DataFrame) -> pd.DataFrame:
        home = df[
            ["Date", "HomeTeam", "FTHG", "FTAG", "HST", "AST", "home_xg", "away_xg"]
        ].copy()
        home.columns = ["Date", "Team", "GF", "GA", "SoT", "SoTAgainst", "xG", "xGA"]
        home["IsHome"] = 1

        away = df[
            ["Date", "AwayTeam", "FTAG", "FTHG", "AST", "HST", "away_xg", "home_xg"]
        ].copy()
        away.columns = ["Date", "Team", "GF", "GA", "SoT", "SoTAgainst", "xG", "xGA"]
        away["IsHome"] = 0

        records = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date"])
        records["Points"] = records.apply(
            lambda r: 3 if r["GF"] > r["GA"] else (1 if r["GF"] == r["GA"] else 0),
            axis=1,
        )
        return records

    def compute_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        records = self._team_match_records(df)
        stat_cols = ["GF", "GA", "SoT", "SoTAgainst", "xG", "xGA"]

        out_parts: list[pd.DataFrame] = []
        for team in records["Team"].unique():
            team_df = records[records["Team"] == team].copy()
            for col in stat_cols:
                team_df[f"avg_{col}"] = (
                    team_df[col].shift(1).rolling(self.window, min_periods=self.min_periods).mean()
                )
            team_df["Form"] = (
                team_df["Points"].shift(1).rolling(self.window, min_periods=self.min_periods).mean()
            )
            out_parts.append(team_df)

        return pd.concat(out_parts, ignore_index=True)

    def build_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        team_stats = self.compute_team_stats(df)
        stat_features = [c for c in team_stats.columns if c.startswith("avg_")] + ["Form"]

        rows: list[dict] = []
        for idx, match in df.iterrows():
            date = match["Date"]
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            home_stats = team_stats[
                (team_stats["Team"] == home)
                & (team_stats["Date"] == date)
                & (team_stats["IsHome"] == 1)
            ]
            away_stats = team_stats[
                (team_stats["Team"] == away)
                & (team_stats["Date"] == date)
                & (team_stats["IsHome"] == 0)
            ]
            if home_stats.empty or away_stats.empty:
                continue

            row: dict = {"match_idx": idx}
            for feat in stat_features:
                h_val = float(home_stats[feat].values[0])
                a_val = float(away_stats[feat].values[0])
                row[f"home_{feat}"] = h_val
                row[f"away_{feat}"] = a_val
                if feat.startswith("avg_"):
                    row[f"diff_{feat}"] = h_val - a_val
            row["diff_Form"] = row["home_Form"] - row["away_Form"]
            rows.append(row)

        features_df = pd.DataFrame(rows).set_index("match_idx")
        featured = df.join(features_df, how="inner")
        featured = featured.dropna(subset=[c for c in features_df.columns if c != "match_idx"])
        return featured


def main() -> None:
    print("STEP 2 (World Cup): building international features...")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    engineer = InternationalFeatureEngineer(window=ROLLING_WINDOW, min_periods=MIN_PERIODS)
    featured = engineer.build_match_features(df)

    print("  Computing ELO (neutral-site home advantage = 0)...")
    elo = FootballELO(k=32, home_advantage=0)
    featured = elo.compute_elo_features(featured)

    print("  Adding H2H features...")
    featured = add_h2h_features(featured, n=5)

    if "is_knockout" not in featured.columns:
        featured["is_knockout"] = 0

    featured.to_csv(OUTPUT_FILE, index=False)
    print(f"OK: saved {len(featured):,} matches -> {OUTPUT_FILE}")
    present = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"  Feature columns present: {len(present)}/{len(FEATURE_COLS)}")


if __name__ == "__main__":
    main()
