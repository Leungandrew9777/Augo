#!/usr/bin/env python3
"""
Step 1: ingest international match data from StatBomb open data (+ optional football-data.org)
and write international_matches_clean.csv.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from statsbombpy import sb

from team_aliases import canonical_name

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(APP_DIR, "international_matches_clean.csv")
EVENT_CACHE_FILE = os.path.join(APP_DIR, ".event_stats_cache.json")

# StatBomb open-data international tournaments with full event coverage.
STATBOMB_COMPETITIONS: list[dict[str, Any]] = [
    {"competition_id": 43, "season_id": 106, "label": "FIFA World Cup 2022"},
    {"competition_id": 43, "season_id": 3, "label": "FIFA World Cup 2018"},
    {"competition_id": 55, "season_id": 282, "label": "UEFA Euro 2024"},
    {"competition_id": 55, "season_id": 43, "label": "UEFA Euro 2020"},
    {"competition_id": 223, "season_id": 282, "label": "Copa America 2024"},
    {"competition_id": 1267, "season_id": 107, "label": "AFCON 2023"},
]

KNOCKOUT_KEYWORDS = (
    "round of",
    "quarter",
    "semi",
    "final",
    "3rd",
    "third",
)


def _is_knockout_stage(stage: str) -> bool:
    s = str(stage or "").lower()
    return any(k in s for k in KNOCKOUT_KEYWORDS)


def _load_event_cache() -> dict[str, dict[str, float | int]]:
    if not os.path.exists(EVENT_CACHE_FILE):
        return {}
    try:
        with open(EVENT_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_event_cache(cache: dict[str, dict[str, float | int]]) -> None:
    tmp = EVENT_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, EVENT_CACHE_FILE)


def _aggregate_match_events(match_id: int, home_team: str, away_team: str) -> dict[str, float | int]:
    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return {
            "HS": 0,
            "AS": 0,
            "HST": 0,
            "AST": 0,
            "home_xg": 0.0,
            "away_xg": 0.0,
        }

    shots["xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)
    shots["on_target"] = shots.get("shot_outcome", pd.Series(dtype=object)).astype(str).str.contains(
        "Goal|Saved", case=False, na=False
    )

    home_mask = shots["team"] == home_team
    away_mask = shots["team"] == away_team

    return {
        "HS": int(home_mask.sum()),
        "AS": int(away_mask.sum()),
        "HST": int((home_mask & shots["on_target"]).sum()),
        "AST": int((away_mask & shots["on_target"]).sum()),
        "home_xg": float(shots.loc[home_mask, "xg"].sum()),
        "away_xg": float(shots.loc[away_mask, "xg"].sum()),
    }


def load_statbomb_matches() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache = _load_event_cache()
    cache_updated = False

    for comp in STATBOMB_COMPETITIONS:
        label = comp["label"]
        print(f"  Loading {label}...")
        matches = sb.matches(competition_id=comp["competition_id"], season_id=comp["season_id"])
        if matches is None or matches.empty:
            continue

        for i, row in matches.iterrows():
            match_id = int(row["match_id"])
            home = canonical_name(str(row["home_team"]))
            away = canonical_name(str(row["away_team"]))
            cache_key = str(match_id)

            if cache_key in cache:
                stats = cache[cache_key]
            else:
                try:
                    stats = _aggregate_match_events(match_id, str(row["home_team"]), str(row["away_team"]))
                    cache[cache_key] = stats
                    cache_updated = True
                except Exception as exc:
                    print(f"    WARN: events failed for match {match_id}: {exc}")
                    stats = {"HS": 0, "AS": 0, "HST": 0, "AST": 0, "home_xg": 0.0, "away_xg": 0.0}

            stage = str(row.get("competition_stage", "") or "")
            rows.append(
                {
                    "Date": pd.to_datetime(row["match_date"]).strftime("%d/%m/%Y"),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": int(row["home_score"]),
                    "FTAG": int(row["away_score"]),
                    "HS": int(stats["HS"]),
                    "AS": int(stats["AS"]),
                    "HST": int(stats["HST"]),
                    "AST": int(stats["AST"]),
                    "home_xg": float(stats["home_xg"]),
                    "away_xg": float(stats["away_xg"]),
                    "competition": label,
                    "stage": stage,
                    "neutral_venue": 1,
                    "is_knockout": int(_is_knockout_stage(stage)),
                    "match_id": match_id,
                }
            )
            if (i + 1) % 20 == 0:
                print(f"    processed {i + 1}/{len(matches)} matches")

    if cache_updated:
        _save_event_cache(cache)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_football_data_matches() -> pd.DataFrame:
    """Optional supplement from football-data.org when FOOTBALL_DATA_API_KEY is set."""
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()

    headers = {"X-Auth-Token": api_key}
    base = "https://api.football-data.org/v4"
    rows: list[dict[str, Any]] = []

    # WC 2022 competition code on football-data.org
    for code, label in [("WC", "FIFA World Cup (football-data.org)")]:
        try:
            resp = requests.get(f"{base}/competitions/{code}/matches", headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"  WARN: football-data.org {code} failed: {exc}")
            continue

        for m in payload.get("matches", []):
            if m.get("status") != "FINISHED":
                continue
            home = canonical_name(m.get("homeTeam", {}).get("name", ""))
            away = canonical_name(m.get("awayTeam", {}).get("name", ""))
            score = m.get("score", {}).get("fullTime", {}) or {}
            hg = score.get("home")
            ag = score.get("away")
            if hg is None or ag is None:
                continue
            rows.append(
                {
                    "Date": pd.to_datetime(m.get("utcDate")).strftime("%d/%m/%Y"),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": int(hg),
                    "FTAG": int(ag),
                    "HS": pd.NA,
                    "AS": pd.NA,
                    "HST": pd.NA,
                    "AST": pd.NA,
                    "home_xg": pd.NA,
                    "away_xg": pd.NA,
                    "competition": label,
                    "stage": str(m.get("stage", "")),
                    "neutral_venue": 1,
                    "is_knockout": int(_is_knockout_stage(str(m.get("stage", "")))),
                    "match_id": f"fd_{m.get('id')}",
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def merge_and_dedupe(statbomb_df: pd.DataFrame, supplement_df: pd.DataFrame) -> pd.DataFrame:
    if statbomb_df.empty and supplement_df.empty:
        return pd.DataFrame()

    parts = [df for df in (statbomb_df, supplement_df) if not df.empty]
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    # Prefer StatBomb rows (with xG) over football-data.org duplicates.
    df["_has_xg"] = df["home_xg"].notna() & df["away_xg"].notna()
    df = df.sort_values(["Date", "_has_xg"], ascending=[True, False])
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    df = df.drop(columns=["_has_xg"], errors="ignore")

    # Result encoding consistent with Augo: H=2, D=1, A=0
    df["FTR"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTHG"] < r["FTAG"] else "D"),
        axis=1,
    )
    df["Result"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})

    return df.sort_values("Date").reset_index(drop=True)


def main() -> None:
    print("STEP 1 (World Cup): loading StatBomb international tournaments...")
    statbomb_df = load_statbomb_matches()
    print(f"  StatBomb rows: {len(statbomb_df):,}")

    print("STEP 1 (World Cup): optional football-data.org supplement...")
    supplement_df = load_football_data_matches()
    if supplement_df.empty:
        print("  No football-data.org rows (missing key or API error).")
    else:
        print(f"  football-data.org rows: {len(supplement_df):,}")

    df = merge_and_dedupe(statbomb_df, supplement_df)
    if df.empty:
        raise SystemExit("No international matches loaded.")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"OK: saved {len(df):,} matches -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
