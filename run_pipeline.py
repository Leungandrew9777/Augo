#!/usr/bin/env python3
"""
run_pipeline.py  —  Augo prediction pipeline
Run ONCE before each gameweek. Output: predictions_cache.json
Usage:
    python run_pipeline.py              # auto-detects next gameweek
    python run_pipeline.py --gw 33      # force a specific matchweek
"""
import argparse, json, os, sys
from datetime import datetime
import joblib, pandas as pd

CACHE_FILE    = "predictions_cache.json"
FIXTURES_FILE = "fixtures.csv"
ELO_FILE      = "premier_league_with_elo_best.csv"
MODEL_FILE    = "xgboost_premier_league_model.pkl"

TEAM_BADGES: dict[str, str] = {
    "Arsenal":                 "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Aston Villa":             "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Bournemouth":             "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Brentford":               "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Brighton & Hove Albion":  "https://resources.premierleague.com/premierleague/badges/t36.png",
    "Burnley":                 "https://resources.premierleague.com/premierleague/badges/t90.png",
    "Chelsea":                 "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Crystal Palace":          "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Everton":                 "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Fulham":                  "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Leeds United":            "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Liverpool":               "https://resources.premierleague.com/premierleague/badges/t14.png",
    "Manchester City":         "https://resources.premierleague.com/premierleague/badges/t43.png",
    "Manchester United":       "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Newcastle":               "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Nottingham Forest":       "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Sunderland":              "https://resources.premierleague.com/premierleague/badges/t56.png",
    "Tottenham Hotspur":       "https://resources.premierleague.com/premierleague/badges/t6.png",
    "West Ham United":         "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Wolverhampton Wanderers": "https://resources.premierleague.com/premierleague/badges/t39.png",
}
FALLBACK_BADGE = "https://resources.premierleague.com/premierleague/badges/t0.png"

FEATURE_COLS = [
    "elo_diff",
    "home_win_rate_5",  "home_win_rate_10",
    "away_win_rate_5",  "away_win_rate_10",
    "home_draw_rate_5", "away_draw_rate_5",
    "h2h_home_win_rate",
    "home_xg_5", "away_xg_5",
    "home_xga_5", "away_xga_5",
    "xg_diff",
]


def compute_current_elo(upcoming: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    latest_elo: dict[str, float] = {}
    for team in pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique():
        m = df_elo[(df_elo["home_team"] == team) | (df_elo["away_team"] == team)]
        if len(m):
            last = m.sort_values("date").iloc[-1]
            latest_elo[team] = (
                last["elo_home_before"] if last["home_team"] == team
                else last["elo_away_before"]
            )
        else:
            latest_elo[team] = 1500.0
    upcoming["elo_home"] = upcoming["home_team"].map(latest_elo).fillna(1500.0)
    upcoming["elo_away"] = upcoming["away_team"].map(latest_elo).fillna(1500.0)
    upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
    return upcoming


def build_features(upcoming: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    for window in [5, 10]:
        upcoming[f"home_win_rate_{window}"] = upcoming["home_team"].apply(
            lambda t: df_elo[df_elo["home_team"] == t].tail(window)["result"].eq("H").mean()
            if len(df_elo[df_elo["home_team"] == t]) > 0 else 0.5
        )
        upcoming[f"home_draw_rate_{window}"] = upcoming["home_team"].apply(
            lambda t: df_elo[df_elo["home_team"] == t].tail(window)["result"].eq("D").mean()
            if len(df_elo[df_elo["home_team"] == t]) > 0 else 0.3
        )
        upcoming[f"away_win_rate_{window}"] = upcoming["away_team"].apply(
            lambda t: df_elo[df_elo["away_team"] == t].tail(window)["result"].eq("A").mean()
            if len(df_elo[df_elo["away_team"] == t]) > 0 else 0.5
        )
        upcoming[f"away_draw_rate_{window}"] = upcoming["away_team"].apply(
            lambda t: df_elo[df_elo["away_team"] == t].tail(window)["result"].eq("D").mean()
            if len(df_elo[df_elo["away_team"] == t]) > 0 else 0.3
        )
    upcoming["h2h_home_win_rate"] = 0.5

    if "home_xg" in df_elo.columns and "away_xg" in df_elo.columns:
        upcoming["home_xg_5"] = upcoming["home_team"].apply(
            lambda t: df_elo[df_elo["home_team"] == t].tail(5)["home_xg"].mean()
            if len(df_elo[df_elo["home_team"] == t]) > 0 else 1.30)
        upcoming["away_xg_5"] = upcoming["away_team"].apply(
            lambda t: df_elo[df_elo["away_team"] == t].tail(5)["away_xg"].mean()
            if len(df_elo[df_elo["away_team"] == t]) > 0 else 1.10)
        upcoming["home_xga_5"] = upcoming["home_team"].apply(
            lambda t: df_elo[df_elo["home_team"] == t].tail(5)["away_xg"].mean()
            if len(df_elo[df_elo["home_team"] == t]) > 0 else 1.10)
        upcoming["away_xga_5"] = upcoming["away_team"].apply(
            lambda t: df_elo[df_elo["away_team"] == t].tail(5)["home_xg"].mean()
            if len(df_elo[df_elo["away_team"] == t]) > 0 else 1.30)
        upcoming["xg_diff"] = upcoming["home_xg_5"] - upcoming["away_xg_5"]
    else:
        upcoming["home_xg_5"] = 1.30
        upcoming["away_xg_5"] = 1.10
        upcoming["home_xga_5"] = 1.10
        upcoming["away_xga_5"] = 1.30
        upcoming["xg_diff"] = 0.20

    return upcoming


def run_model(upcoming: pd.DataFrame, model) -> pd.DataFrame:
    probs = model.predict_proba(upcoming[FEATURE_COLS])
    upcoming["prob_home"] = probs[:, 0]
    upcoming["prob_draw"] = probs[:, 1]
    upcoming["prob_away"] = probs[:, 2]
    upcoming["fair_odds_home"] = 1 / upcoming["prob_home"]
    upcoming["fair_odds_draw"] = 1 / upcoming["prob_draw"]
    upcoming["fair_odds_away"] = 1 / upcoming["prob_away"]
    upcoming["disp_odds_home"] = upcoming["fair_odds_home"].map(lambda v: f"{v:.3g}")
    upcoming["disp_odds_draw"] = upcoming["fair_odds_draw"].map(lambda v: f"{v:.3g}")
    upcoming["disp_odds_away"] = upcoming["fair_odds_away"].map(lambda v: f"{v:.3g}")
    upcoming["disp_prob_home"] = upcoming["prob_home"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_prob_draw"] = upcoming["prob_draw"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_prob_away"] = upcoming["prob_away"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_elo_diff"]  = upcoming["elo_diff"].map(lambda v: f"{v:+.0f}")
    upcoming["badge_home"]     = upcoming["home_team"].map(lambda t: TEAM_BADGES.get(t, FALLBACK_BADGE))
    upcoming["badge_away"]     = upcoming["away_team"].map(lambda t: TEAM_BADGES.get(t, FALLBACK_BADGE))
    upcoming["chart_label"]    = upcoming.apply(
        lambda r: r["home_team"][:3].upper() + " v " + r["away_team"][:3].upper(), axis=1
    )
    upcoming["model_pick"] = upcoming.apply(
        lambda r: max(
            [("H", r["prob_home"]), ("D", r["prob_draw"]), ("A", r["prob_away"])],
            key=lambda x: x[1],
        )[0], axis=1,
    )
    return upcoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=None)
    args = parser.parse_args()

    for path, hint in [
        (MODEL_FILE,    "Run FeatureEng.py first."),
        (ELO_FILE,      "Run ELO.py first."),
        (FIXTURES_FILE, "fixtures.csv must have: matchweek, date, home_team, away_team"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"❌  {path} not found — {hint}")

    print("Loading model and ELO data …")
    model  = joblib.load(MODEL_FILE)
    df_elo = pd.read_csv(ELO_FILE)
    df_elo["date"] = pd.to_datetime(df_elo["date"])

    df_fix = pd.read_csv(FIXTURES_FILE)
    df_fix["date"] = pd.to_datetime(df_fix["date"], errors="coerce").dt.normalize()
    df_fix = df_fix.dropna(subset=["date", "home_team", "away_team"])

    today  = pd.Timestamp.today().normalize()
    future = df_fix[df_fix["date"] >= today].sort_values("date")
    if future.empty:
        sys.exit("❌  No upcoming fixtures found in fixtures.csv.")

    # Detect gameweek column (your CSV uses "matchweek")
    gw_col = next((c for c in ("matchweek", "gameweek") if c in future.columns), None)

    if args.gw is not None:
        if gw_col is None:
            sys.exit("❌  --gw specified but no matchweek/gameweek column found.")
        selected = df_fix[df_fix[gw_col] == args.gw].sort_values("date")
        if selected.empty:
            sys.exit(f"❌  No fixtures found for matchweek {args.gw}.")
        gw_label = f"GW{args.gw}"
    elif gw_col and future[gw_col].notna().any():
        next_gw  = future.iloc[0][gw_col]

        # Optional prompt: allow choosing a later upcoming gameweek.
        # (Keeps --gw for scripted/non-interactive usage.)
        chosen_gw = next_gw
        if sys.stdin.isatty():
            try:
                available = (
                    future[gw_col]
                    .dropna()
                    .astype(int)
                    .sort_values()
                    .unique()
                    .tolist()
                )
                available_after = [g for g in available if g >= int(next_gw)]
            except Exception:
                available_after = []

            if available_after:
                print(f"Auto-detected current gameweek: GW{int(next_gw)}")
                print("Upcoming gameweeks:", ", ".join(f"GW{g}" for g in available_after[:12]) +
                      (" …" if len(available_after) > 12 else ""))
                raw = input("Enter a later GW number to run (or press Enter to keep current): ").strip()
                if raw:
                    try:
                        gw_int = int(raw)
                        if gw_int < int(next_gw):
                            print(f"⚠️  GW{gw_int} is before current GW{int(next_gw)}; using current.")
                        elif gw_int not in available_after:
                            print(f"⚠️  GW{gw_int} not found in upcoming fixtures; using current.")
                        else:
                            chosen_gw = gw_int
                    except ValueError:
                        print("⚠️  Invalid input; using current.")

        # Important: select from ALL fixtures so we show the full gameweek,
        # not just matches on/after today.
        selected = df_fix[df_fix[gw_col] == chosen_gw].sort_values("date")
        gw_label = f"GW{int(chosen_gw)}"
    else:
        start    = future["date"].min()
        selected = future[future["date"] <= start + pd.Timedelta(days=3)]
        gw_label = f"Next ({start.strftime('%b %d')})"

    print(f"Gameweek  : {gw_label}  ({len(selected)} fixtures)")

    known = set(pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique())
    for _, row in selected.iterrows():
        for team in (row["home_team"], row["away_team"]):
            if team not in known:
                print(f"⚠️   Unknown team '{team}' — ELO defaults to 1500")

    upcoming = selected[["date", "home_team", "away_team"]].copy().reset_index(drop=True)
    upcoming = compute_current_elo(upcoming, df_elo)
    upcoming = build_features(upcoming, df_elo)
    upcoming = run_model(upcoming, model)

    records: list[dict] = []
    for i, row in upcoming.iterrows():
        rec = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):  rec[k] = str(v.date())
            elif hasattr(v, "item"):     rec[k] = v.item()
            else:                        rec[k] = v
        rec["match_idx"] = int(i)
        records.append(rec)

    cache = {
        "generated_at": datetime.now().isoformat(),
        "gameweek":     gw_label,
        "predictions":  records,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n{'─'*72}")
    print(f"{'HOME':<26}  {'AWAY':<26}  H%    D%    A%   Pick")
    print(f"{'─'*72}")
    for r in records:
        print(f"{r['home_team']:<26}  {r['away_team']:<26}  "
              f"{r['prob_home']*100:4.1f}  {r['prob_draw']*100:4.1f}  "
              f"{r['prob_away']*100:4.1f}  [{r['model_pick']}]")
    print(f"{'─'*72}")
    print(f"\n✅  Predictions cached → {CACHE_FILE}")
    print(f"   Now launch:  reflex run")


if __name__ == "__main__":
    main()