import reflex as rx
import pandas as pd
import joblib
import difflib
from datetime import datetime


class State(rx.State):
    """Main app state"""
    fixtures: list[dict] = [
        {"date": "2026-04-11", "home_team": "West Ham United", "away_team": "Wolverhampton Wanderers"},
        {"date": "2026-04-11", "home_team": "Arsenal", "away_team": "Bournemouth"},
        {"date": "2026-04-11", "home_team": "Brentford", "away_team": "Everton"},
        {"date": "2026-04-11", "home_team": "Burnley", "away_team": "Brighton & Hove Albion"},
        {"date": "2026-04-12", "home_team": "Liverpool", "away_team": "Fulham"},
        {"date": "2026-04-12", "home_team": "Crystal Palace", "away_team": "Newcastle"},
        {"date": "2026-04-12", "home_team": "Nottingham Forest", "away_team": "Aston Villa"},
        {"date": "2026-04-12", "home_team": "Sunderland", "away_team": "Tottenham Hotspur"},
        {"date": "2026-04-12", "home_team": "Chelsea", "away_team": "Manchester City"},
        {"date": "2026-04-14", "home_team": "Manchester United", "away_team": "Leeds United"}
    ]
    predictions: list[dict] = []
    current_tab: str = "predictions"

    def load_predictions(self):
        """Load model + ELO and calculate predictions"""
        try:
            model = joblib.load("xgboost_premier_league_model.pkl")
            df_elo = pd.read_csv("premier_league_with_elo_best.csv")
            df_elo["date"] = pd.to_datetime(df_elo["date"])

            upcoming = pd.DataFrame(self.fixtures)
            upcoming["date"] = pd.to_datetime(upcoming["date"])

            # Full ELO + feature logic (same as before)
            upcoming = compute_current_elo(upcoming, df_elo)
            upcoming = predict_upcoming(upcoming, df_elo, model)

            self.predictions = upcoming.to_dict("records")
            rx.toast.success("Predictions updated!", position="top-center")
        except Exception as e:
            rx.toast.error(f"Error: {str(e)}")

    def add_empty_fixture(self):
        self.fixtures.append({"date": "2026-04-15", "home_team": "", "away_team": ""})

    def delete_fixture(self, index: int):
        self.fixtures.pop(index)


# Helper functions (same logic as Streamlit version)
def compute_current_elo(upcoming: pd.DataFrame, df_elo: pd.DataFrame):
    latest_elo = {}
    for team in pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique():
        team_matches = df_elo[(df_elo["home_team"] == team) | (df_elo["away_team"] == team)]
        if len(team_matches) > 0:
            last = team_matches.sort_values("date").iloc[-1]
            latest_elo[team] = last["elo_home_before"] if last["home_team"] == team else last["elo_away_before"]
        else:
            latest_elo[team] = 1500.0
    upcoming["elo_home"] = upcoming["home_team"].map(latest_elo)
    upcoming["elo_away"] = upcoming["away_team"].map(latest_elo)
    upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
    return upcoming


def predict_upcoming(upcoming: pd.DataFrame, df_elo: pd.DataFrame, model):
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

    X = upcoming[feature_cols]
    probs = model.predict_proba(X)
    upcoming["prob_home"] = probs[:, 0]
    upcoming["prob_draw"] = probs[:, 1]
    upcoming["prob_away"] = probs[:, 2]
    upcoming["fair_odds_home"] = 1 / upcoming["prob_home"]
    upcoming["fair_odds_draw"] = 1 / upcoming["prob_draw"]
    upcoming["fair_odds_away"] = 1 / upcoming["prob_away"]
    return upcoming


def index():
    return rx.center(
        rx.vstack(
            # Clean header (no extra branding)
            rx.heading("Premier League Predictor", size="8", color="#FF4B4B", text_align="center"),
            rx.text("XGBoost + ELO • April 2026", color="#B0B0B0", text_align="center", margin_bottom="1rem"),

            # Bottom navigation style tabs
            rx.tabs(
                rx.tab("📊 Predictions", value="predictions"),
                rx.tab("✏️ Edit Fixtures", value="edit"),
                value=State.current_tab,
                on_change=State.set_current_tab,
                width="100%",
                variant="soft",
                color_scheme="red",
            ),

            # Predictions tab
            rx.cond(
                State.current_tab == "predictions",
                rx.vstack(
                    rx.button("🔄 Refresh Predictions", on_click=State.load_predictions, color_scheme="red",
                              width="100%"),
                    rx.table(
                        headers=["Home", "Away", "H %", "D %", "A %", "Fair H", "Fair D", "Fair A"],
                        data=[
                            [
                                p["home_team"], p["away_team"],
                                f"{p['prob_home']:.1%}", f"{p['prob_draw']:.1%}", f"{p['prob_away']:.1%}",
                                f"{p['fair_odds_home']:.2f}", f"{p['fair_odds_draw']:.2f}", f"{p['fair_odds_away']:.2f}"
                            ] for p in State.predictions
                        ],
                        style={
                            "background_color": "#1A1A1A",
                            "color": "white",
                            "border_radius": "8px",
                            "overflow": "hidden"
                        }
                    ),
                    align_items="stretch",
                    width="100%",
                    spacing="4"
                ),
                # Edit fixtures tab
                rx.vstack(
                    rx.button("➕ Add Fixture", on_click=State.add_empty_fixture, color_scheme="green", width="100%"),
                    rx.foreach(
                        State.fixtures,
                        lambda f, i: rx.hstack(
                            rx.input(value=f["date"], on_change=lambda v: State.fixtures[i]["date"] = v, width = "30%"),
    rx.input(value=f["home_team"], on_change=lambda v: State.fixtures[i]["home_team"] = v, width = "35%"),
    rx.input(value=f["away_team"], on_change=lambda v: State.fixtures[i]["away_team"] = v, width = "35%"),
    rx.button("🗑️", on_click=lambda: State.delete_fixture(i), color_scheme="red", size="sm"),
    align_items = "center",
    width = "100%"
    )
    ),
    rx.button("🔄 Use these fixtures", on_click=State.load_predictions, color_scheme="red", width="100%"),
    align_items = "stretch",
    width = "100%",
    spacing = "3"
    )
    ),
    spacing = "6",
    padding = "4",
    min_height = "100vh",
    background_color = "#0E1117",
    color = "white"
    )
    )

    # Create the app
    app = rx.App()
    app.add_page(index, route="/")
    app.run()