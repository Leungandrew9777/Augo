import reflex as rx
import pandas as pd
import joblib
import json
import os
from typing import Any

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────

PL_TEAMS: list[str] = sorted([
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton & Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Leeds United", "Liverpool", "Manchester City",
    "Manchester United", "Newcastle", "Nottingham Forest", "Sunderland",
    "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
])

TEAM_BADGES: dict[str, str] = {
    "Arsenal":                    "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Aston Villa":                "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Bournemouth":                "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Brentford":                  "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Brighton & Hove Albion":     "https://resources.premierleague.com/premierleague/badges/t36.png",
    "Burnley":                    "https://resources.premierleague.com/premierleague/badges/t90.png",
    "Chelsea":                    "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Crystal Palace":             "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Everton":                    "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Fulham":                     "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Leeds United":               "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Liverpool":                  "https://resources.premierleague.com/premierleague/badges/t14.png",
    "Manchester City":            "https://resources.premierleague.com/premierleague/badges/t43.png",
    "Manchester United":          "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Newcastle":                  "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Nottingham Forest":          "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Sunderland":                 "https://resources.premierleague.com/premierleague/badges/t56.png",
    "Tottenham Hotspur":          "https://resources.premierleague.com/premierleague/badges/t6.png",
    "West Ham United":            "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Wolverhampton Wanderers":    "https://resources.premierleague.com/premierleague/badges/t39.png",
}

FALLBACK_BADGE = "https://resources.premierleague.com/premierleague/badges/t0.png"

FALLBACK_FIXTURES: list[dict] = [
    {"idx": 0, "date": "2026-04-11", "home_team": "West Ham United",     "away_team": "Wolverhampton Wanderers"},
    {"idx": 1, "date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 2, "date": "2026-04-11", "home_team": "Brentford",           "away_team": "Everton"},
    {"idx": 3, "date": "2026-04-11", "home_team": "Burnley",             "away_team": "Brighton & Hove Albion"},
    {"idx": 4, "date": "2026-04-12", "home_team": "Liverpool",           "away_team": "Fulham"},
    {"idx": 5, "date": "2026-04-12", "home_team": "Crystal Palace",      "away_team": "Newcastle"},
    {"idx": 6, "date": "2026-04-12", "home_team": "Nottingham Forest",   "away_team": "Aston Villa"},
    {"idx": 7, "date": "2026-04-12", "home_team": "Sunderland",          "away_team": "Tottenham Hotspur"},
    {"idx": 8, "date": "2026-04-12", "home_team": "Chelsea",             "away_team": "Manchester City"},
    {"idx": 9, "date": "2026-04-14", "home_team": "Manchester United",   "away_team": "Leeds United"},
]

from typing import TypedDict

class MatchDict(TypedDict):
    home_team: str
    away_team: str
    badge_home: str
    badge_away: str
    disp_odds_home: str
    disp_odds_draw: str
    disp_odds_away: str
    disp_prob_home: str
    disp_prob_draw: str
    disp_prob_away: str
    prob_home: float
    prob_draw: float
    prob_away: float
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    actual: str
    user_pick: str
    model_pick: str

class GWDict(TypedDict):
    idx: int
    gw: str
    date: str
    matches: list[MatchDict]
    model_accuracy: str
    user_accuracy: str
    pnl: str
    pnl_positive: bool

class FixtureDict(TypedDict):
    idx: int
    date: str
    home_team: str
    away_team: str

class PredictionDict(TypedDict):
    match_idx: int
    home_team: str
    away_team: str
    badge_home: str
    badge_away: str
    disp_odds_home: str
    disp_odds_draw: str
    disp_odds_away: str
    disp_prob_home: str
    disp_prob_draw: str
    disp_prob_away: str
    prob_home: float
    prob_draw: float
    prob_away: float
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    disp_elo_diff: str
    chart_label: str
    model_pick: str

class ChartBarDict(TypedDict):
    label: str
    home: float
    draw: float
    away: float

class EloChartDict(TypedDict):
    label: str
    elo_diff: float
    elo_positive: bool

# ── State ─────────────────────────────────────────────────────────────────────

class State(rx.State):
    current_tab: str = "home"

    # Matchweek
    fixtures: list[dict[str, Any]] = FALLBACK_FIXTURES
    predictions: list[dict[str, Any]] = []
    gameweek_label: str = "GW32"

    # User picks for current GW (parallel to predictions list)
    user_picks: list[str] = []   # "" / "H" / "D" / "A"

    # Custom predictor
    custom_home: str = PL_TEAMS[0]
    custom_away: str = PL_TEAMS[1]
    custom_result: list[dict[str, Any]] = []

    # Insights
    safe_picks: list[dict[str, Any]] = []
    coin_flips: list[dict[str, Any]] = []
    top_pick: dict[str, Any] = {}
    underdog: dict[str, Any] = {}
    win_prob_chart: list[dict[str, Any]] = []
    elo_chart: list[dict[str, Any]] = []

    # History
    history: list[dict[str, Any]] = []
    history_selected: int = -1

    # ── Computed vars ─────────────────────────────────────────────────────────

    @rx.var
    def predictions_with_picks(self) -> list[dict[str, Any]]:
        """Merge predictions with current user picks for display."""
        result = []
        for i, p in enumerate(self.predictions):
            pick = self.user_picks[i] if i < len(self.user_picks) else ""
            result.append({**p, "user_pick": pick})
        return result

    @rx.var
    def picks_count(self) -> int:
        return sum(1 for p in self.user_picks if p != "")

    @rx.var
    def total_fixtures(self) -> int:
        return len(self.predictions)

    @rx.var
    def all_picked(self) -> bool:
        return len(self.user_picks) > 0 and all(p != "" for p in self.user_picks)

    @rx.var
    def picks_agree_count(self) -> int:
        """Count how many user picks agree with model picks."""
        count = 0
        for i, p in enumerate(self.predictions):
            if i < len(self.user_picks) and self.user_picks[i] != "":
                if self.user_picks[i] == p.get("model_pick", ""):
                    count += 1
        return count

    @rx.var
    def selected_gw_entry(self) -> dict[str, Any]:
        if self.history_selected < 0 or self.history_selected >= len(self.history):
            return {}
        return self.history[self.history_selected]

    @rx.var
    def selected_gw_matches(self) -> list[dict[str, Any]]:
        if self.history_selected < 0 or self.history_selected >= len(self.history):
            return []
        matches = self.history[self.history_selected]["matches"]
        return [dict(m, match_idx=i) for i, m in enumerate(matches)]

    # ── Fixtures ──────────────────────────────────────────────────────────────

    def _reindex(self):
        for i, f in enumerate(self.fixtures):
            f["idx"] = i

    def load_fixtures_from_csv(self):
        # Use an absolute path so running from a different CWD
        # (e.g. during export/dev server) still reads the intended file.
        path = os.path.join(APP_DIR, "fixtures.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df = df.copy()
                # Expect ISO dates: YYYY-MM-DD
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
                df = df.dropna(subset=["date", "home_team", "away_team"])
                today = pd.Timestamp.today().normalize()

                # Accept either "gameweek" or "matchweek" as the GW column name
                gw_col = next((c for c in ["gameweek", "matchweek"] if c in df.columns and df[c].notna().any()), None)

                if gw_col:
                    # Find the current or next gameweek:
                    # - If any matches in a GW are still today/future, that GW is active
                    # - This keeps mid-gameweek refreshes showing the full GW (including
                    #   already-played Saturday games alongside upcoming Sunday ones)
                    future = df[df["date"] >= today].sort_values("date")
                    if future.empty:
                        self.fixtures = FALLBACK_FIXTURES
                        return
                    current_gw = future.iloc[0][gw_col]
                    # Pull ALL fixtures for that GW from the full df, not just future ones
                    selected = df[df[gw_col] == current_gw].sort_values("date")
                    self.gameweek_label = f"GW{int(current_gw)}" if str(current_gw).isdigit() else f"{current_gw}"
                else:
                    # No GW column — fall back to a 4-day window around the next fixture
                    future = df[df["date"] >= today].sort_values("date")
                    if future.empty:
                        self.fixtures = FALLBACK_FIXTURES
                        return
                    start = future["date"].min()
                    end = start + pd.Timedelta(days=3)
                    selected = df[(df["date"] >= start) & (df["date"] <= end)]
                    self.gameweek_label = f"Next ({start.strftime('%b %d')})"
                rows: list[dict[str, Any]] = []
                for i, row in enumerate(selected.itertuples(index=False), start=0):
                    rows.append({
                        "idx": i,
                        "date": str(getattr(row, "date").date()),
                        "home_team": str(getattr(row, "home_team")),
                        "away_team": str(getattr(row, "away_team")),
                    })
                if rows:
                    self.fixtures = rows
            except Exception:
                self.fixtures = FALLBACK_FIXTURES
        else:
            self.fixtures = FALLBACK_FIXTURES

    def add_fixture(self):
        self.fixtures.append({"idx": len(self.fixtures), "date": "", "home_team": "", "away_team": ""})

    def delete_fixture(self, idx: int):
        self.fixtures = [f for f in self.fixtures if f["idx"] != idx]
        self._reindex()

    def update_date(self, idx: int, value: str):
        self.fixtures[idx]["date"] = value

    def update_home_fixture(self, idx: int, value: str):
        self.fixtures[idx]["home_team"] = value

    def update_away_fixture(self, idx: int, value: str):
        self.fixtures[idx]["away_team"] = value

    # ── ML pipeline ──────────────────────────────────────────────────────────

    def _load_model_and_elo(self):
        model = joblib.load("xgboost_premier_league_model.pkl")
        df_elo = pd.read_csv("premier_league_with_elo_best.csv")
        df_elo["date"] = pd.to_datetime(df_elo["date"])
        return model, df_elo

    def _compute_current_elo(self, upcoming: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
        latest_elo: dict = {}
        for team in pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique():
            m = df_elo[(df_elo["home_team"] == team) | (df_elo["away_team"] == team)]
            if len(m) > 0:
                last = m.sort_values("date").iloc[-1]
                latest_elo[team] = last["elo_home_before"] if last["home_team"] == team else last["elo_away_before"]
            else:
                latest_elo[team] = 1500.0
        upcoming["elo_home"] = upcoming["home_team"].map(latest_elo).fillna(1500.0)
        upcoming["elo_away"] = upcoming["away_team"].map(latest_elo).fillna(1500.0)
        upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
        return upcoming

    def _predict_upcoming(self, upcoming: pd.DataFrame, df_elo: pd.DataFrame, model) -> pd.DataFrame:
        for window in [5, 10]:
            upcoming[f"home_win_rate_{window}"] = upcoming["home_team"].apply(
                lambda t: df_elo[df_elo["home_team"] == t].tail(window)["result"].eq("H").mean()
                if len(df_elo[df_elo["home_team"] == t]) > 0 else 0.5)
            upcoming[f"home_draw_rate_{window}"] = upcoming["home_team"].apply(
                lambda t: df_elo[df_elo["home_team"] == t].tail(window)["result"].eq("D").mean()
                if len(df_elo[df_elo["home_team"] == t]) > 0 else 0.3)
            upcoming[f"away_win_rate_{window}"] = upcoming["away_team"].apply(
                lambda t: df_elo[df_elo["away_team"] == t].tail(window)["result"].eq("A").mean()
                if len(df_elo[df_elo["away_team"] == t]) > 0 else 0.5)
            upcoming[f"away_draw_rate_{window}"] = upcoming["away_team"].apply(
                lambda t: df_elo[df_elo["away_team"] == t].tail(window)["result"].eq("D").mean()
                if len(df_elo[df_elo["away_team"] == t]) > 0 else 0.3)
        upcoming["h2h_home_win_rate"] = 0.5

        feature_cols = ["elo_diff", "home_win_rate_5", "home_win_rate_10",
                        "away_win_rate_5", "away_win_rate_10",
                        "home_draw_rate_5", "away_draw_rate_5", "h2h_home_win_rate"]
        probs = model.predict_proba(upcoming[feature_cols])

        upcoming["prob_home"]      = probs[:, 0]
        upcoming["prob_draw"]      = probs[:, 1]
        upcoming["prob_away"]      = probs[:, 2]
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
            lambda r: r["home_team"][:3].upper() + " v " + r["away_team"][:3].upper(), axis=1)

        # Model's predicted outcome per match
        upcoming["model_pick"] = upcoming.apply(
            lambda r: max(
                [("H", r["prob_home"]), ("D", r["prob_draw"]), ("A", r["prob_away"])],
                key=lambda x: x[1]
            )[0], axis=1
        )
        return upcoming

    def _compute_insights(self, df: pd.DataFrame):
        records = df.to_dict("records")
        for r in records:
            probs = {"H": r["prob_home"], "D": r["prob_draw"], "A": r["prob_away"]}
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            r["pred"] = pred
            r["pred_prob"] = probs[pred]
            if pred == "H":
                r["pred_name"] = r["home_team"]
                r["pred_badge"] = r["badge_home"]
                r["pred_disp_prob"] = r["disp_prob_home"]
                r["pred_disp_odds"] = r["disp_odds_home"]
                r["pred_fair_odds"] = r["fair_odds_home"]
            elif pred == "A":
                r["pred_name"] = r["away_team"]
                r["pred_badge"] = r["badge_away"]
                r["pred_disp_prob"] = r["disp_prob_away"]
                r["pred_disp_odds"] = r["disp_odds_away"]
                r["pred_fair_odds"] = r["fair_odds_away"]
            else:
                r["pred_name"] = "Draw"
                r["pred_badge"] = ""
                r["pred_disp_prob"] = r["disp_prob_draw"]
                r["pred_disp_odds"] = r["disp_odds_draw"]
                r["pred_fair_odds"] = r["fair_odds_draw"]

        self.safe_picks = [r for r in records if r["pred_prob"] > 0.65]
        # Coin flip: no clear favourite — best outcome probability under 50%
        self.coin_flips = [
            r for r in records
            if r["pred_prob"] < 0.50
        ]
        self.top_pick = max(records, key=lambda r: r["pred_prob"], default={})
        underdogs = [r for r in records if r.get("pred") in ("H", "A")]
        self.underdog = min(underdogs, key=lambda r: r["pred_prob"], default={})
        self.win_prob_chart = [
            {"label": r["chart_label"], "home": round(r["prob_home"]*100,1),
             "draw": round(r["prob_draw"]*100,1), "away": round(r["prob_away"]*100,1)}
            for r in records
        ]
        self.elo_chart = [
            {"label": r["chart_label"], "elo_diff": round(r["elo_diff"], 0),
             "elo_positive": bool(r["elo_diff"] >= 0)}
            for r in records
        ]

    def load_predictions(self):
        self.load_fixtures_from_csv()
        try:
            model, df_elo = self._load_model_and_elo()
            raw = [{k: v for k, v in f.items() if k != "idx"} for f in self.fixtures]
            upcoming = pd.DataFrame(raw)
            upcoming["date"] = pd.to_datetime(upcoming["date"])
            upcoming = self._compute_current_elo(upcoming, df_elo)
            upcoming = self._predict_upcoming(upcoming, df_elo, model)
            self.predictions = upcoming.to_dict("records")
            # Add match index to each prediction
            for i in range(len(self.predictions)):
                self.predictions[i]["match_idx"] = i
            # Reset user picks
            self.user_picks = [""] * len(self.predictions)
            self._compute_insights(upcoming)
        except Exception as e:
            return rx.toast.error(f"Prediction error: {e}")

    # ── Custom predictor ──────────────────────────────────────────────────────

    def run_custom_prediction(self):
        if self.custom_home == self.custom_away:
            return rx.toast.error("Pick two different teams.")
        try:
            model, df_elo = self._load_model_and_elo()
            df = pd.DataFrame([{
                "date": str(pd.Timestamp.today().date()),
                "home_team": self.custom_home,
                "away_team": self.custom_away,
            }])
            df["date"] = pd.to_datetime(df["date"])
            df = self._compute_current_elo(df, df_elo)
            df = self._predict_upcoming(df, df_elo, model)
            self.custom_result = df.to_dict("records")
        except Exception as e:
            return rx.toast.error(f"Prediction error: {e}")

    # ── User picks ────────────────────────────────────────────────────────────

    def set_user_pick(self, match_idx: int, pick: str):
        """Set the user's pick for a specific match."""
        picks = list(self.user_picks)
        while len(picks) <= match_idx:
            picks.append("")
        picks[match_idx] = pick
        self.user_picks = picks

    def lock_in_picks(self):
        """Save current GW predictions + user picks to history."""
        if not self.predictions:
            return rx.toast.error("No predictions loaded.")
        if not self.all_picked:
            return rx.toast.error("Pick an outcome for every match first.")

        matches = []
        for i, p in enumerate(self.predictions):
            user_pick = self.user_picks[i] if i < len(self.user_picks) else ""
            model_pick = p.get("model_pick", "")
            matches.append({
                "home_team":      p["home_team"],
                "away_team":      p["away_team"],
                "badge_home":     p["badge_home"],
                "badge_away":     p["badge_away"],
                "disp_odds_home": p["disp_odds_home"],
                "disp_odds_draw": p["disp_odds_draw"],
                "disp_odds_away": p["disp_odds_away"],
                "disp_prob_home": p["disp_prob_home"],
                "disp_prob_draw": p["disp_prob_draw"],
                "disp_prob_away": p["disp_prob_away"],
                "prob_home":      p["prob_home"],
                "prob_draw":      p["prob_draw"],
                "prob_away":      p["prob_away"],
                "fair_odds_home": p["fair_odds_home"],
                "fair_odds_draw": p["fair_odds_draw"],
                "fair_odds_away": p["fair_odds_away"],
                "actual":         "",
                "user_pick":      user_pick,
                "model_pick":     model_pick,
            })

        entry = {
            "idx":            len(self.history),
            "gw":             self.gameweek_label,
            "date":           self.fixtures[0]["date"] if self.fixtures else "",
            "matches":        matches,
            "model_accuracy": "",
            "user_accuracy":  "",
            "pnl":            "",
            "pnl_positive":   False,
        }
        self.history.append(entry)
        agree = self.picks_agree_count
        total = self.total_fixtures
        return rx.toast.success(
            f"{self.gameweek_label} locked in! You agreed with model on {agree}/{total} matches."
        )

    # ── History ───────────────────────────────────────────────────────────────

    def select_history(self, idx: int):
        self.history_selected = idx

    def back_to_history_list(self):
        self.history_selected = -1

    def set_actual_result(self, gw_idx: int, match_idx: int, result: str):
        """Mark actual result and recalculate both user and model accuracy."""
        self.history[gw_idx]["matches"][match_idx]["actual"] = result
        self._recalculate_history(gw_idx)

    def _recalculate_history(self, gw_idx: int):
        matches = self.history[gw_idx]["matches"]
        completed = [m for m in matches if m["actual"] in ("H", "D", "A")]
        if not completed:
            self.history[gw_idx]["model_accuracy"] = ""
            self.history[gw_idx]["user_accuracy"] = ""
            self.history[gw_idx]["pnl"] = ""
            return

        model_correct = 0
        user_correct = 0
        pnl = 0.0

        for m in completed:
            actual = m["actual"]
            model_pick = m.get("model_pick", "")
            user_pick = m.get("user_pick", "")

            if model_pick == actual:
                model_correct += 1
            if user_pick == actual:
                user_correct += 1

            # P&L based on model (flat £1 stake on model's pick)
            odds_map = {
                "H": m["fair_odds_home"],
                "D": m["fair_odds_draw"],
                "A": m["fair_odds_away"],
            }
            if model_pick and model_pick == actual:
                pnl += odds_map.get(model_pick, 1.0) - 1.0
            elif model_pick:
                pnl -= 1.0

        n = len(completed)
        model_acc = model_correct / n * 100
        user_acc = user_correct / n * 100

        self.history[gw_idx]["model_accuracy"] = f"{model_acc:.0f}%  ({model_correct}/{n})"
        self.history[gw_idx]["user_accuracy"]  = f"{user_acc:.0f}%  ({user_correct}/{n})"
        sign = "+" if pnl >= 0 else ""
        self.history[gw_idx]["pnl"] = f"{sign}{pnl:.2f} u"
        self.history[gw_idx]["pnl_positive"] = bool(pnl >= 0)


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def badge_img(url, size: str = "28px") -> rx.Component:
    return rx.image(src=url, width=size, height=size,
                    style={"object_fit": "contain", "flex_shrink": "0"})


def odds_col(label: str, odds, prob, color: str) -> rx.Component:
    return rx.vstack(
        rx.text(label, color="white",  font_size="0.65em", letter_spacing="0.1em", font_weight="600"),
        rx.text(odds,  color="white", font_size="1.5em",  font_weight="700",       line_height="1"),
        rx.text(prob,  color="white", font_size="0.74em"),
        spacing="1",
        align="center",
        width="33%",
    )


def match_card(p: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                badge_img(p["badge_home"]),
                rx.text(p["home_team"], font_weight="600", font_size="0.82em",
                        color="#d0d0d0", flex="1", text_align="right"),
                rx.text("v", color="#2a2a2a", font_size="0.7em", padding_x="8px"),
                rx.text(p["away_team"], font_weight="600", font_size="0.82em",
                        color="#d0d0d0", flex="1"),
                badge_img(p["badge_away"]),
                width="100%",
                align="center",
            ),
            rx.box(height="1px", width="100%", background_color="#1e1e1e"),
            rx.hstack(
                odds_col("H", p["disp_odds_home"], p["disp_prob_home"], "#4CAF50"),
                odds_col("D", p["disp_odds_draw"], p["disp_prob_draw"], "#FFC107"),
                odds_col("A", p["disp_odds_away"], p["disp_prob_away"], "#F44336"),
                width="100%",
                justify="between",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        width="100%",
        padding="13px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def section_label(text: str) -> rx.Component:
    return rx.text(text, color="#555", font_size="0.68em",
                   letter_spacing="0.12em", font_weight="700",
                   padding_bottom="4px")


# ── Home tab ──────────────────────────────────────────────────────────────────

def home_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() > 0,
            rx.vstack(
                rx.hstack(
                    section_label(State.gameweek_label),
                    rx.spacer(),
                    rx.text("Model Predictions", color="#444", font_size="0.7em"),
                    width="100%",
                    align="center",
                    padding_bottom="6px",
                ),
                rx.foreach(State.predictions.to(list[dict[str, Any]]), match_card),
                width="100%",
                spacing="2",
            ),
            rx.box(
                rx.text("Loading predictions…", color="#444", font_size="0.88em"),
                text_align="center", padding_y="48px", width="100%",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Predictor tab ─────────────────────────────────────────────────────────────

def pick_outcome_btn(label: str, pick_dict: dict, outcome: str, color: str) -> rx.Component:
    """A single H / D / A pick button for a fixture."""
    is_selected = pick_dict["user_pick"] == outcome
    is_model    = pick_dict["model_pick"] == outcome
    return rx.box(
        rx.vstack(
            rx.text(label, font_size="0.75em", font_weight="700",
                    color=rx.cond(is_selected, "#0E1117", "#888"),
                    line_height="1"),
            rx.cond(
                is_model,
                rx.text("◆", font_size="0.5em",
                        color=rx.cond(is_selected, "#0E1117", color),
                        line_height="1"),
                rx.box(height="8px"),
            ),
            spacing="1",
            align="center",
        ),
        on_click=State.set_user_pick(pick_dict["match_idx"], outcome),
        background_color=rx.cond(is_selected, color, "#1a1a1a"),
        border=rx.cond(is_model,
                       rx.cond(is_selected, f"2px solid {color}", f"2px solid {color}"),
                       "2px solid #2a2a2a"),
        border_radius="6px",
        padding="6px 0",
        width="58px",
        text_align="center",
        cursor="pointer",
        transition="all 0.15s ease",
    )


def pick_card(p: dict) -> rx.Component:
    """Single fixture card with user pick buttons and model indicator."""
    return rx.box(
        rx.vstack(
            # Teams row
            rx.hstack(
                badge_img(p["badge_home"], "22px"),
                rx.text(p["home_team"], font_size="0.78em", font_weight="600",
                        color="#d0d0d0", flex="1", text_align="right", no_of_lines=1),
                rx.text("v", color="#2a2a2a", font_size="0.65em", padding_x="6px"),
                rx.text(p["away_team"], font_size="0.78em", font_weight="600",
                        color="#d0d0d0", flex="1", no_of_lines=1),
                badge_img(p["badge_away"], "22px"),
                width="100%",
                align="center",
            ),
            rx.box(height="1px", width="100%", background_color="#1e1e1e"),
            # Pick buttons row
            rx.hstack(
                pick_outcome_btn("H", p, "H", "#4CAF50"),
                pick_outcome_btn("D", p, "D", "#FFC107"),
                pick_outcome_btn("A", p, "A", "#F44336"),
                rx.spacer(),
                # Probability hint
                rx.vstack(
                    rx.text(p["disp_prob_home"], color="#4CAF50", font_size="0.65em"),
                    rx.text(p["disp_prob_draw"], color="#FFC107", font_size="0.65em"),
                    rx.text(p["disp_prob_away"], color="#F44336", font_size="0.65em"),
                    spacing="0",
                    align="end",
                ),
                width="100%",
                align="center",
                spacing="2",
            ),
            # Legend hint
            rx.text("◆ = model pick", color="#333", font_size="0.6em", align_self="end"),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def predictor_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() == 0,
            rx.box(
                rx.text("Predictions not loaded yet.", color="#444", font_size="0.85em"),
                text_align="center", padding_y="48px", width="100%",
            ),
            rx.vstack(
                # Header with GW label + picks counter
                rx.hstack(
                    rx.vstack(
                        rx.text(State.gameweek_label, color="white",
                                font_size="1em", font_weight="700"),
                        rx.text("Your Picks vs Model", color="#555",
                                font_size="0.68em", letter_spacing="0.08em"),
                        spacing="0",
                        align="start",
                    ),
                    rx.spacer(),
                    # Picks progress pill
                    rx.box(
                        rx.hstack(
                            rx.text(State.picks_count.to_string(), color="white",
                                    font_size="0.9em", font_weight="700"),
                            rx.text("/" + State.total_fixtures.to_string(),
                                    color="#555", font_size="0.9em"),
                            spacing="0",
                        ),
                        padding="4px 10px",
                        background_color="#1a1a1a",
                        border="1px solid #2a2a2a",
                        border_radius="20px",
                    ),
                    width="100%",
                    align="center",
                    padding_bottom="8px",
                ),

                # Match pick cards
                rx.foreach(
                    State.predictions_with_picks.to(list[dict[str, Any]]),
                    pick_card,
                ),

                # Agreement summary (shown once picks start)
                rx.cond(
                    State.picks_count > 0,
                    rx.box(
                        rx.hstack(
                            rx.icon("handshake", size=14, color="#888"),
                            rx.text(
                                "You agree with model on ",
                                rx.text.span(State.picks_agree_count.to_string(),
                                             color="white", font_weight="700"),
                                rx.text.span(" / " + State.picks_count.to_string()),
                                " picks so far",
                                color="#555",
                                font_size="0.72em",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        padding="8px 12px",
                        background_color="#141414",
                        border_radius="6px",
                        border="1px solid #1e1e1e",
                        width="100%",
                    ),
                    rx.box(),
                ),

                # Lock In button
                rx.box(
                    rx.hstack(
                        rx.icon("lock", size=15, color=rx.cond(State.all_picked, "white", "#444")),
                        rx.text(
                            "Lock In Picks & Save",
                            color=rx.cond(State.all_picked, "white", "#444"),
                            font_size="0.88em", font_weight="600",
                        ),
                        spacing="2",
                        align="center",
                        justify="center",
                    ),
                    on_click=State.lock_in_picks,
                    width="100%",
                    padding_y="12px",
                    background_color=rx.cond(State.all_picked, "#FF4B4B", "#1a1a1a"),
                    border="1px solid",
                    border_color=rx.cond(State.all_picked, "#FF4B4B", "#2a2a2a"),
                    border_radius="8px",
                    text_align="center",
                    cursor=rx.cond(State.all_picked, "pointer", "default"),
                    transition="all 0.2s ease",
                ),

                width="100%",
                spacing="2",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Insights tab ──────────────────────────────────────────────────────────────

def insight_card(title: str, body: rx.Component) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(title, color="#888", font_size="0.68em",
                    letter_spacing="0.1em", font_weight="700"),
            body,
            spacing="2",
            align="start",
            width="100%",
        ),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def safe_pick_row(p: dict) -> rx.Component:
    return rx.hstack(
        badge_img(p["badge_home"], "20px"),
        rx.text(p["home_team"], color="#d0d0d0", font_size="0.8em", flex="1",
                text_align="right", no_of_lines=1, style={"min_width": "0"}),
        rx.text("v", color="#333", font_size="0.7em"),
        rx.text(p["away_team"], color="#d0d0d0", font_size="0.8em", flex="1",
                text_align="left", no_of_lines=1, style={"min_width": "0"}),
        badge_img(p["badge_away"], "20px"),
        rx.spacer(),
        rx.text(p["pred_name"], color="#aaa", font_size="0.72em", font_weight="600"),
        rx.text(p["pred_disp_prob"], color="#4CAF50", font_size="0.8em", font_weight="600"),
        width="100%",
        align="center",
        spacing="2",
    )


def coin_flip_row(p: dict) -> rx.Component:
    return rx.hstack(
        badge_img(p["badge_home"], "20px"),
        rx.text(p["home_team"], color="#d0d0d0", font_size="0.8em", flex="1"),
        rx.text("v", color="#333", font_size="0.7em"),
        rx.text(p["away_team"], color="#d0d0d0", font_size="0.8em", flex="1"),
        badge_img(p["badge_away"], "20px"),
        width="100%",
        align="center",
        spacing="2",
    )


def chart_bar_row(item: dict) -> rx.Component:
    return rx.vstack(
        rx.text(item["label"], color="#555", font_size="0.65em", letter_spacing="0.04em"),
        rx.hstack(
            rx.box(
                rx.text(item["home"], color="white", font_size="0.6em", padding_x="3px"),
                background_color="#4CAF50",
                border_radius="2px 0 0 2px",
                height="14px",
                width=item["home"].to_string() + "%",
                display="flex", align_items="center",
            ),
            rx.box(
                rx.text(item["draw"], color="white", font_size="0.6em", padding_x="3px"),
                background_color="#FFC107",
                height="14px",
                width=item["draw"].to_string() + "%",
                display="flex", align_items="center",
            ),
            rx.box(
                rx.text(item["away"], color="white", font_size="0.6em", padding_x="3px"),
                background_color="#F44336",
                border_radius="0 2px 2px 0",
                height="14px",
                width=item["away"].to_string() + "%",
                display="flex", align_items="center",
            ),
            spacing="0",
            width="100%",
        ),
        spacing="1",
        width="100%",
    )


def elo_bar_row(item: dict) -> rx.Component:
    return rx.hstack(
        rx.text(item["label"], color="#555", font_size="0.65em",
                width="70px", flex_shrink="0"),
        rx.box(
            rx.text(item["elo_diff"], color="white", font_size="0.6em", padding_x="4px"),
            background_color=rx.cond(item["elo_positive"], "#4CAF50", "#F44336"),
            border_radius="3px",
            height="14px",
            min_width="28px",
            display="flex",
            align_items="center",
        ),
        width="100%",
        align="center",
        spacing="2",
    )


def insights_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() == 0,
            rx.box(
                rx.text("Run predictions on the Home tab first.", color="#444", font_size="0.85em"),
                text_align="center", padding_y="48px", width="100%",
            ),
            rx.vstack(
                insight_card(
                    "SAFE PICKS  >65%",
                    rx.cond(
                        State.safe_picks.length() > 0,
                        rx.vstack(rx.foreach(State.safe_picks.to(list[dict[str, Any]]), safe_pick_row),
                                  spacing="2", width="100%"),
                        rx.text("No match clears 65% this week.", color="#555", font_size="0.8em"),
                    ),
                ),
                insight_card(
                    "COIN FLIPS  (no clear favourite  <50%)",
                    rx.cond(
                        State.coin_flips.length() > 0,
                        rx.vstack(rx.foreach(State.coin_flips.to(list[dict[str, Any]]), coin_flip_row),
                                  spacing="2", width="100%"),
                        rx.text("No true coin-flip matches this week.", color="#555", font_size="0.8em"),
                    ),
                ),
                insight_card(
                    "WIN PROBABILITIES  ( H / D / A )",
                    rx.vstack(
                        rx.foreach(State.win_prob_chart.to(list[dict[str, Any]]), chart_bar_row),
                        spacing="3", width="100%",
                    ),
                ),
                insight_card(
                    "ELO ADVANTAGE  (home positive = home favoured)",
                    rx.vstack(
                        rx.foreach(State.elo_chart.to(list[dict[str, Any]]), elo_bar_row),
                        spacing="3", width="100%",
                    ),
                ),
                width="100%",
                spacing="3",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── History tab ───────────────────────────────────────────────────────────────

def pick_chip(label: str, color: str, is_correct: bool, has_actual: bool) -> rx.Component:
    """Small chip showing a pick with correct/incorrect indicator."""
    return rx.hstack(
        rx.text(label, font_size="0.72em", font_weight="700",
                color=rx.cond(has_actual,
                              rx.cond(is_correct, "#0E1117", "white"),
                              "white")),
        rx.cond(
            has_actual,
            rx.cond(
                is_correct,
                rx.icon("check", size=10, color="#0E1117"),
                rx.icon("x", size=10, color="white"),
            ),
            rx.box(),
        ),
        spacing="1",
        align="center",
        padding="3px 8px",
        background_color=rx.cond(has_actual,
                                  rx.cond(is_correct, color, "#3a1a1a"),
                                  "#1e1e1e"),
        border="1px solid",
        border_color=rx.cond(has_actual,
                              rx.cond(is_correct, color, "#F44336"),
                              color),
        border_radius="20px",
    )


def actual_btn(gw_idx: int, match_idx: int, label: str, outcome: str, current_actual: str) -> rx.Component:
    is_set = current_actual == outcome
    color_map = {"H": "#4CAF50", "D": "#FFC107", "A": "#F44336"}
    c = color_map.get(outcome, "#888")
    return rx.box(
        rx.text(label, font_size="0.7em", font_weight="700",
                color=rx.cond(is_set, "#0E1117", "#666")),
        on_click=State.set_actual_result(gw_idx, match_idx, outcome),
        background_color=rx.cond(is_set, c, "#1a1a1a"),
        border=f"1px solid {c}",
        border_radius="4px",
        padding="4px 10px",
        cursor="pointer",
    )


def history_match_detail_row(m: dict) -> rx.Component:
    """Detailed row in expanded GW view: user pick, model pick, actual input."""
    gw_idx   = State.history_selected
    has_actual = m["actual"] != ""
    user_correct  = m["user_pick"]  == m["actual"]
    model_correct = m["model_pick"] == m["actual"]

    return rx.box(
        rx.vstack(
            # Teams
            rx.hstack(
                badge_img(m["badge_home"], "18px"),
                rx.text(m["home_team"], color="#bbb", font_size="0.75em",
                        flex="1", text_align="right", no_of_lines=1),
                rx.text("v", color="#333", font_size="0.65em"),
                rx.text(m["away_team"], color="#bbb", font_size="0.75em",
                        flex="1", no_of_lines=1),
                badge_img(m["badge_away"], "18px"),
                width="100%",
                align="center",
                spacing="2",
            ),
            # Picks + actual
            rx.hstack(
                # User pick chip
                rx.vstack(
                    rx.text("YOU", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    pick_chip(m["user_pick"], "#6C63FF", user_correct & has_actual, has_actual),
                    spacing="1", align="center",
                ),
                rx.text("vs", color="#2a2a2a", font_size="0.65em"),
                # Model pick chip
                rx.vstack(
                    rx.text("MODEL", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    pick_chip(m["model_pick"], "#FF9800", model_correct & has_actual, has_actual),
                    spacing="1", align="center",
                ),
                rx.spacer(),
                # Actual result buttons
                rx.vstack(
                    rx.text("RESULT", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    rx.hstack(
                        actual_btn(gw_idx, m["match_idx"], "H", "H", m["actual"]),
                        actual_btn(gw_idx, m["match_idx"], "D", "D", m["actual"]),
                        actual_btn(gw_idx, m["match_idx"], "A", "A", m["actual"]),
                        spacing="1",
                    ),
                    spacing="1",
                    align="end",
                ),
                width="100%",
                align="end",
                spacing="2",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="10px 12px",
        background_color="#0f0f0f",
        border_radius="6px",
        border="1px solid #1a1a1a",
    )


def history_gw_card(entry: dict) -> rx.Component:
    """Summary card for a saved GW in the history list."""
    has_results = entry["model_accuracy"] != ""
    return rx.box(
        rx.vstack(
            # GW + date row
            rx.hstack(
                rx.text(entry["gw"], color="white", font_size="0.85em", font_weight="700"),
                rx.text(entry["date"], color="#444", font_size="0.72em"),
                rx.spacer(),
                rx.icon("chevron-right", size=14, color="#333"),
                width="100%",
                align="center",
            ),
            # Accuracy chips row
            rx.cond(
                has_results,
                rx.hstack(
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background_color="#6C63FF"),
                        rx.text("You: ", color="#555", font_size="0.7em"),
                        rx.text(entry["user_accuracy"], color="#6C63FF",
                                font_size="0.72em", font_weight="600"),
                        spacing="1", align="center",
                    ),
                    rx.text("|", color="#222", font_size="0.7em"),
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background_color="#FF9800"),
                        rx.text("Model: ", color="#555", font_size="0.7em"),
                        rx.text(entry["model_accuracy"], color="#FF9800",
                                font_size="0.72em", font_weight="600"),
                        spacing="1", align="center",
                    ),
                    rx.spacer(),
                    rx.text(entry["pnl"],
                            color=rx.cond(entry["pnl_positive"], "#4CAF50", "#F44336"),
                            font_size="0.7em", font_weight="600"),
                    width="100%",
                    align="center",
                    spacing="2",
                ),
                rx.text("Enter results to see accuracy", color="#333", font_size="0.68em"),
            ),
            spacing="2",
            width="100%",
        ),
        on_click=State.select_history(entry["idx"]),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
        cursor="pointer",
    )


def history_detail_view() -> rx.Component:
    """Expanded view for a selected GW."""
    entry = State.selected_gw_entry
    return rx.vstack(
        # Back button + header
        rx.hstack(
            rx.box(
                rx.hstack(
                    rx.icon("chevron-left", size=14, color="#888"),
                    rx.text("Back", color="#888", font_size="0.75em"),
                    spacing="1", align="center",
                ),
                on_click=State.back_to_history_list,
                cursor="pointer",
            ),
            rx.spacer(),
            rx.vstack(
                rx.text(entry["gw"], color="white", font_size="0.9em", font_weight="700"),
                rx.text(entry["date"], color="#444", font_size="0.68em"),
                spacing="0", align="end",
            ),
            width="100%",
            align="center",
            padding_bottom="4px",
        ),
        rx.box(height="1px", width="100%", background_color="#1e1e1e"),

        # Score summary (if any results entered)
        rx.cond(
            entry["model_accuracy"] != "",
            rx.hstack(
                # User score box
                rx.box(
                    rx.vstack(
                        rx.text("YOU", color="#6C63FF", font_size="0.6em",
                                letter_spacing="0.1em", font_weight="700"),
                        rx.text(entry["user_accuracy"], color="white",
                                font_size="0.82em", font_weight="700"),
                        spacing="1", align="center",
                    ),
                    flex="1",
                    padding="10px 8px",
                    background_color="#12101f",
                    border="1px solid #2a2560",
                    border_radius="8px",
                    text_align="center",
                ),
                # vs divider
                rx.text("vs", color="#2a2a2a", font_size="0.75em"),
                # Model score box
                rx.box(
                    rx.vstack(
                        rx.text("MODEL", color="#FF9800", font_size="0.6em",
                                letter_spacing="0.1em", font_weight="700"),
                        rx.text(entry["model_accuracy"], color="white",
                                font_size="0.82em", font_weight="700"),
                        spacing="1", align="center",
                    ),
                    flex="1",
                    padding="10px 8px",
                    background_color="#1a1000",
                    border="1px solid #3a2800",
                    border_radius="8px",
                    text_align="center",
                ),
                width="100%",
                align="center",
                spacing="2",
            ),
            rx.box(
                rx.text("Tap H / D / A on each match to enter results",
                        color="#333", font_size="0.72em"),
                text_align="center", padding_y="6px", width="100%",
            ),
        ),

        # Match rows
        rx.vstack(
            rx.foreach(
                State.selected_gw_matches.to(list[dict[str, Any]]),
                history_match_detail_row,
            ),
            width="100%",
            spacing="2",
        ),
        width="100%",
        spacing="3",
    )


def history_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.history_selected >= 0,
            # Detail view for selected GW
            history_detail_view(),
            # List view
            rx.cond(
                State.history.length() == 0,
                rx.box(
                    rx.vstack(
                        rx.icon("clock", size=32, color="#222"),
                        rx.text("No history yet.", color="#444", font_size="0.85em"),
                        rx.text("Head to Predictor, make your picks, and lock them in.",
                                color="#333", font_size="0.75em", text_align="center"),
                        spacing="2",
                        align="center",
                    ),
                    text_align="center", padding_y="48px", width="100%",
                ),
                rx.vstack(
                    section_label("GAMEWEEK HISTORY"),
                    rx.foreach(State.history.to(list[dict[str, Any]]), history_gw_card),
                    width="100%",
                    spacing="2",
                ),
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Navbar ────────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("home",      "Home",      "home"),
    ("predictor", "Predictor", "crosshair"),
    ("insights",  "Insights",  "bar-chart-2"),
    ("history",   "History",   "clock"),
]


def nav_btn(tab: str, label: str, icon: str) -> rx.Component:
    active = State.current_tab == tab
    return rx.box(
        rx.vstack(
            rx.icon(icon, size=17, color=rx.cond(active, "white", "#444")),
            rx.text(label, font_size="0.6em", letter_spacing="0.03em",
                    font_weight=rx.cond(active, "600", "400"),
                    color=rx.cond(active, "white", "#444")),
            spacing="1",
            align="center",
        ),
        on_click=State.set_current_tab(tab),
        width="25%",
        height="56px",
        display="flex",
        align_items="center",
        justify_content="center",
        background_color=rx.cond(active, "#1a0000", "#0E1117"),
        border_top=rx.cond(active, "2px solid #FF4B4B", "2px solid transparent"),
        cursor="pointer",
    )


def navbar() -> rx.Component:
    return rx.hstack(
        *[nav_btn(tab, label, icon) for tab, label, icon in NAV_ITEMS],
        width="100%",
        spacing="0",
        position="fixed",
        bottom="0",
        left="0",
        right="0",
        z_index="1000",
        background_color="#0E1117",
        border_top="1px solid #1a1a1a",
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def index() -> rx.Component:
    content = rx.cond(
        State.current_tab == "home",
        home_tab(),
        rx.cond(
            State.current_tab == "predictor",
            predictor_tab(),
            rx.cond(
                State.current_tab == "insights",
                insights_tab(),
                rx.cond(
                    State.current_tab == "history",
                    history_tab(),
                    rx.box(),
                ),
            ),
        ),
    )

    return rx.box(
        rx.vstack(
            rx.box(
                rx.heading("Augo", size="6", color="white", font_weight="700"),
                rx.text("XGBoost + ELO · 2025/26", color="#333",
                        font_size="0.7em", letter_spacing="0.06em"),
                padding_x="16px",
                padding_top="18px",
                padding_bottom="10px",
            ),
            rx.box(height="1px", width="100%", background_color="#1a1a1a"),
            rx.box(
                content,
                width="100%",
                padding_x="16px",
                padding_bottom="80px",
                padding_top="12px",
            ),
            spacing="0",
            width="100%",
            min_height="100vh",
            background_color="#0E1117",
        ),
        navbar(),
        width="100%",
        background_color="#0E1117",
    )


app = rx.App(
    style={
        "background_color": "#0E1117",
        "color": "white",
        "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }
)
app.add_page(index, route="/", on_load=State.load_predictions)