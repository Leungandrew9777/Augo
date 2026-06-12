#!/usr/bin/env python3
"""
Weekly / per-round inference for World Cup fixtures -> predictions_cache.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

APP_DIR = os.path.dirname(os.path.abspath(__file__))

_wc_fe_path = os.path.join(APP_DIR, "feature_engineering.py")
_wc_fe_spec = importlib.util.spec_from_file_location("wc_feature_engineering", _wc_fe_path)
if _wc_fe_spec is None or _wc_fe_spec.loader is None:
    raise ImportError(f"Could not load {_wc_fe_path}")
_wc_fe = importlib.util.module_from_spec(_wc_fe_spec)
_wc_fe_spec.loader.exec_module(_wc_fe)
FEATURE_COLS = _wc_fe.FEATURE_COLS

_aliases_path = os.path.join(APP_DIR, "team_aliases.py")
_aliases_spec = importlib.util.spec_from_file_location("wc_team_aliases", _aliases_path)
if _aliases_spec is None or _aliases_spec.loader is None:
    raise ImportError(f"Could not load {_aliases_path}")
_aliases = importlib.util.module_from_spec(_aliases_spec)
_aliases_spec.loader.exec_module(_aliases)
canonical_name = _aliases.canonical_name
fixture_lookup_key = _aliases.fixture_lookup_key

load_dotenv()

CACHE_FILE = os.path.join(APP_DIR, "predictions_cache.json")
FIXTURES_FILE = os.path.join(APP_DIR, "fixtures.csv")
FEATURES_FILE = os.path.join(APP_DIR, "international_with_features.csv")
CLEAN_FILE = os.path.join(APP_DIR, "international_matches_clean.csv")
MODEL_FILE = os.path.join(APP_DIR, "world_cup_xgb_model.pkl")
HISTORY_DIR = os.path.join(APP_DIR, "predictions_history")

FALLBACK_BADGE = "https://flagcdn.com/w80/xx.png"


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    if "Date" in out.columns and "date" not in out.columns:
        rename["Date"] = "date"
    if "HomeTeam" in out.columns and "home_team" not in out.columns:
        rename["HomeTeam"] = "home_team"
    if "AwayTeam" in out.columns and "away_team" not in out.columns:
        rename["AwayTeam"] = "away_team"
    if rename:
        out = out.rename(columns=rename)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out["home_team"] = out["home_team"].map(canonical_name)
    out["away_team"] = out["away_team"].map(canonical_name)
    if "FTR" in out.columns:
        out["result"] = out["FTR"]
    elif "Result" in out.columns:
        out["result"] = out["Result"].map({0: "A", 1: "D", 2: "H"})
    return out.sort_values("date")


def _team_rolling_at_date(
    history: pd.DataFrame,
    team: str,
    fixture_date: pd.Timestamp,
    *,
    window: int = 10,
    min_periods: int = 1,
) -> dict[str, float]:
    prior = history[
        ((history["home_team"] == team) | (history["away_team"] == team))
        & (history["date"] < fixture_date)
    ].sort_values("date").tail(window)

    if len(prior) < min_periods:
        return {}

    rows: list[dict[str, float]] = []
    for _, m in prior.iterrows():
        if m["home_team"] == team:
            rows.append(
                {
                    "GF": float(m["FTHG"]),
                    "GA": float(m["FTAG"]),
                    "SoT": float(m.get("HST", 0) or 0),
                    "SoTAgainst": float(m.get("AST", 0) or 0),
                    "xG": float(m.get("home_xg", 0) or 0),
                    "xGA": float(m.get("away_xg", 0) or 0),
                    "Points": 3.0 if m["FTHG"] > m["FTAG"] else (1.0 if m["FTHG"] == m["FTAG"] else 0.0),
                }
            )
        else:
            rows.append(
                {
                    "GF": float(m["FTAG"]),
                    "GA": float(m["FTHG"]),
                    "SoT": float(m.get("AST", 0) or 0),
                    "SoTAgainst": float(m.get("HST", 0) or 0),
                    "xG": float(m.get("away_xg", 0) or 0),
                    "xGA": float(m.get("home_xg", 0) or 0),
                    "Points": 3.0 if m["FTAG"] > m["FTHG"] else (1.0 if m["FTAG"] == m["FTHG"] else 0.0),
                }
            )

    stats = pd.DataFrame(rows).mean()
    return {
        "avg_GF": float(stats["GF"]),
        "avg_GA": float(stats["GA"]),
        "avg_SoT": float(stats["SoT"]),
        "avg_xG": float(stats["xG"]),
        "avg_xGA": float(stats["xGA"]),
        "Form": float(stats["Points"]),
    }


def _h2h_features(history: pd.DataFrame, home: str, away: str, fixture_date: pd.Timestamp) -> dict[str, float]:
    prior = history[
        (history["date"] < fixture_date)
        & (
            ((history["home_team"] == home) & (history["away_team"] == away))
            | ((history["home_team"] == away) & (history["away_team"] == home))
        )
    ].tail(5)
    if prior.empty:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_total_goals_avg": 2.5}

    wins = draws = total_goals = 0
    for _, p in prior.iterrows():
        gh, ga = int(p["FTHG"]), int(p["FTAG"])
        total_goals += gh + ga
        if p["home_team"] == home:
            wins += gh > ga
            draws += gh == ga
        else:
            wins += ga > gh
            draws += gh == ga
    n = len(prior)
    return {
        "h2h_home_wins": wins / n,
        "h2h_draws": draws / n,
        "h2h_total_goals_avg": total_goals / n,
    }


def _latest_elo(df_features: pd.DataFrame, team: str, side: str) -> float:
    col = f"elo_{side}_before"
    if col not in df_features.columns:
        return 1500.0
    matches = df_features[(df_features["home_team"] == team) | (df_features["away_team"] == team)]
    if matches.empty:
        return 1500.0
    last = matches.sort_values("date").iloc[-1]
    if last["home_team"] == team:
        return float(last["elo_home_before"])
    return float(last["elo_away_before"])


def build_upcoming_features(upcoming: pd.DataFrame, history: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    history = _normalize_history_df(history)
    df_features = _normalize_history_df(df_features)
    medians = df_features[[c for c in FEATURE_COLS if c in df_features.columns]].median()

    rows: list[dict] = []
    for _, fix in upcoming.iterrows():
        home = canonical_name(str(fix["home_team"]))
        away = canonical_name(str(fix["away_team"]))
        date = pd.to_datetime(fix["date"], errors="coerce")

        home_roll = _team_rolling_at_date(history, home, date)
        away_roll = _team_rolling_at_date(history, away, date)
        h2h = _h2h_features(history, home, away, date)

        row = {
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_avg_xG": home_roll.get("avg_xG", medians.get("home_avg_xG", 1.2)),
            "away_avg_xG": away_roll.get("avg_xG", medians.get("away_avg_xG", 1.2)),
            "home_avg_xGA": home_roll.get("avg_xGA", medians.get("home_avg_xGA", 1.2)),
            "away_avg_xGA": away_roll.get("avg_xGA", medians.get("away_avg_xGA", 1.2)),
            "home_Form": home_roll.get("Form", medians.get("home_Form", 1.5)),
            "away_Form": away_roll.get("Form", medians.get("away_Form", 1.5)),
            "home_avg_GF": home_roll.get("avg_GF", medians.get("home_avg_GF", 1.2)),
            "away_avg_GF": away_roll.get("avg_GF", medians.get("away_avg_GF", 1.2)),
            "home_avg_GA": home_roll.get("avg_GA", medians.get("home_avg_GA", 1.2)),
            "away_avg_GA": away_roll.get("avg_GA", medians.get("away_avg_GA", 1.2)),
            "home_avg_SoT": home_roll.get("avg_SoT", medians.get("home_avg_SoT", 4.0)),
            "away_avg_SoT": away_roll.get("avg_SoT", medians.get("away_avg_SoT", 4.0)),
            "is_knockout": int(fix.get("is_knockout", 0) or 0),
            **h2h,
        }
        row["diff_avg_xG"] = row["home_avg_xG"] - row["away_avg_xG"]
        row["diff_avg_xGA"] = row["home_avg_xGA"] - row["away_avg_xGA"]
        row["diff_Form"] = row["home_Form"] - row["away_Form"]
        row["elo_home"] = _latest_elo(df_features, home, "home")
        row["elo_away"] = _latest_elo(df_features, away, "away")
        row["elo_diff"] = row["elo_home"] - row["elo_away"]
        rows.append(row)

    return pd.DataFrame(rows)


def run_model(upcoming: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    X = upcoming[feature_cols].fillna(upcoming[feature_cols].median())
    probs = model.predict_proba(X)
    upcoming = upcoming.copy()
    upcoming["prob_away"] = probs[:, 0]
    upcoming["prob_draw"] = probs[:, 1]
    upcoming["prob_home"] = probs[:, 2]
    upcoming["fair_odds_home"] = 1 / upcoming["prob_home"].clip(lower=1e-6)
    upcoming["fair_odds_draw"] = 1 / upcoming["prob_draw"].clip(lower=1e-6)
    upcoming["fair_odds_away"] = 1 / upcoming["prob_away"].clip(lower=1e-6)
    upcoming["disp_prob_home"] = upcoming["prob_home"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_prob_draw"] = upcoming["prob_draw"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_prob_away"] = upcoming["prob_away"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_odds_home"] = upcoming["fair_odds_home"].map(lambda v: f"{v:.2f}")
    upcoming["disp_odds_draw"] = upcoming["fair_odds_draw"].map(lambda v: f"{v:.2f}")
    upcoming["disp_odds_away"] = upcoming["fair_odds_away"].map(lambda v: f"{v:.2f}")
    upcoming["disp_elo_diff"] = upcoming["elo_diff"].map(lambda v: f"{v:+.0f}")
    upcoming["badge_home"] = FALLBACK_BADGE
    upcoming["badge_away"] = FALLBACK_BADGE
    upcoming["chart_label"] = upcoming.apply(
        lambda r: r["home_team"][:3].upper() + " v " + r["away_team"][:3].upper(),
        axis=1,
    )
    upcoming["model_pick"] = upcoming.apply(
        lambda r: max(
            [("H", r["prob_home"]), ("D", r["prob_draw"]), ("A", r["prob_away"])],
            key=lambda x: x[1],
        )[0],
        axis=1,
    )
    return upcoming


def predict_fixtures(upcoming: pd.DataFrame) -> pd.DataFrame:
    """Public helper for app live inference."""
    artifact = joblib.load(MODEL_FILE)
    history = pd.read_csv(CLEAN_FILE)
    df_features = pd.read_csv(FEATURES_FILE)
    featured = build_upcoming_features(upcoming, history, df_features)
    return run_model(featured, artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup prediction pipeline")
    parser.add_argument("--round", type=int, default=None, help="Force tournament round number")
    args = parser.parse_args()

    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"Model not found: {MODEL_FILE}. Run train_xgb.py first.")

    fixtures = pd.read_csv(FIXTURES_FILE)
    fixtures["date"] = pd.to_datetime(fixtures["date"], dayfirst=True, errors="coerce")
    fixtures = fixtures.dropna(subset=["date", "home_team", "away_team"])
    fixtures["home_team"] = fixtures["home_team"].map(fixture_lookup_key)
    fixtures["away_team"] = fixtures["away_team"].map(fixture_lookup_key)

    round_col = next((c for c in ["round", "gameweek", "matchweek"] if c in fixtures.columns), None)
    if args.round is not None and round_col:
        fixtures = fixtures[fixtures[round_col] == args.round]
    elif round_col:
        today = pd.Timestamp.today().normalize()
        future = fixtures[fixtures["date"] >= today].sort_values("date")
        if not future.empty:
            current_round = future.iloc[0][round_col]
            fixtures = fixtures[fixtures[round_col] == current_round]

    if fixtures.empty:
        raise SystemExit("No fixtures to predict.")

    upcoming = fixtures.rename(columns={"home_team": "home_team", "away_team": "away_team"}).copy()
    predictions_df = predict_fixtures(upcoming)

    round_label = f"R{int(args.round)}" if args.round is not None else (
        f"R{int(fixtures[round_col].iloc[0])}" if round_col else "Next"
    )

    export_df = predictions_df.copy()
    if "date" in export_df.columns:
        export_df["date"] = pd.to_datetime(export_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    cache = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gameweek": round_label,
        "competition": "world_cup",
        "predictions": export_df.to_dict("records"),
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    os.makedirs(HISTORY_DIR, exist_ok=True)
    archive_path = os.path.join(HISTORY_DIR, f"{round_label}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"OK: wrote {len(predictions_df)} predictions -> {CACHE_FILE}")
    print(f"     archived -> {archive_path}")


if __name__ == "__main__":
    main()
