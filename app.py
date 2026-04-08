import streamlit as st
import pandas as pd
import joblib
import difflib
from pathlib import Path

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== HIDE DEFAULT STREAMLIT UI ======================
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {background-color: #0E1117;}
    .css-1d391kg {padding-top: 1rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>
        🚀 Premier League Matchweek Predictor
    </h1>
    <p style='text-align: center; color: #B0B0B0; font-size: 18px;'>
        Powered by your XGBoost + ELO model • Updated April 2026
    </p>
    <hr style='margin: 0 0 2rem 0;'>
""", unsafe_allow_html=True)


# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    model = joblib.load("xgboost_premier_league_model.pkl")
    df_elo = pd.read_csv("premier_league_with_elo_best.csv")
    df_elo['date'] = pd.to_datetime(df_elo['date'])
    return model, df_elo


model, df_elo = load_data()


# ====================== FUNCTIONS (same as before, just cleaned) ======================
def validate_fixtures(upcoming, df_elo):
    valid_teams = set(pd.concat([df_elo['home_team'], df_elo['away_team']]).unique())
    errors = []
    for _, row in upcoming.iterrows():
        home, away = row['home_team'], row['away_team']
        if home not in valid_teams:
            sug = difflib.get_close_matches(home, valid_teams, n=1, cutoff=0.6)
            errors.append(f"❌ Home team '{home}' → Did you mean '{sug[0]}'?" if sug else f"❌ Home team '{home}'")
        if away not in valid_teams:
            sug = difflib.get_close_matches(away, valid_teams, n=1, cutoff=0.6)
            errors.append(f"❌ Away team '{away}' → Did you mean '{sug[0]}'?" if sug else f"❌ Away team '{away}'")
    if errors:
        st.error("\n".join(errors))
        st.stop()
    return True


def compute_current_elo(upcoming, df_elo):
    latest_elo = {}
    for team in pd.concat([df_elo['home_team'], df_elo['away_team']]).unique():
        team_matches = df_elo[(df_elo['home_team'] == team) | (df_elo['away_team'] == team)]
        if len(team_matches) > 0:
            last = team_matches.sort_values('date').iloc[-1]
            latest_elo[team] = last['elo_home_before'] if last['home_team'] == team else last['elo_away_before']
        else:
            latest_elo[team] = 1500.0
    upcoming['elo_home'] = upcoming['home_team'].map(latest_elo)
    upcoming['elo_away'] = upcoming['away_team'].map(latest_elo)
    upcoming['elo_diff'] = upcoming['elo_home'] - upcoming['elo_away']
    return upcoming


def predict_upcoming(upcoming, df_elo):
    upcoming = compute_current_elo(upcoming, df_elo)

    for window in [5, 10]:
        upcoming[f'home_win_rate_{window}'] = upcoming['home_team'].apply(
            lambda t: df_elo[df_elo['home_team'] == t].tail(window)['result'].eq('H').mean()
            if len(df_elo[df_elo['home_team'] == t]) > 0 else 0.5)
        upcoming[f'home_draw_rate_{window}'] = upcoming['home_team'].apply(
            lambda t: df_elo[df_elo['home_team'] == t].tail(window)['result'].eq('D').mean()
            if len(df_elo[df_elo['home_team'] == t]) > 0 else 0.3)
        upcoming[f'away_win_rate_{window}'] = upcoming['away_team'].apply(
            lambda t: df_elo[df_elo['away_team'] == t].tail(window)['result'].eq('A').mean()
            if len(df_elo[df_elo['away_team'] == t]) > 0 else 0.5)
        upcoming[f'away_draw_rate_{window}'] = upcoming['away_team'].apply(
            lambda t: df_elo[df_elo['away_team'] == t].tail(window)['result'].eq('D').mean()
            if len(df_elo[df_elo['away_team'] == t]) > 0 else 0.3)

    upcoming['h2h_home_win_rate'] = 0.5

    feature_cols = ['elo_diff', 'home_win_rate_5', 'home_win_rate_10',
                    'away_win_rate_5', 'away_win_rate_10',
                    'home_draw_rate_5', 'away_draw_rate_5', 'h2h_home_win_rate']

    X = upcoming[feature_cols]
    probs = model.predict_proba(X)

    upcoming['prob_home'] = probs[:, 0]
    upcoming['prob_draw'] = probs[:, 1]
    upcoming['prob_away'] = probs[:, 2]
    upcoming['fair_odds_home'] = 1 / upcoming['prob_home']
    upcoming['fair_odds_draw'] = 1 / upcoming['prob_draw']
    upcoming['fair_odds_away'] = 1 / upcoming['prob_away']

    return upcoming


# ====================== FIXTURES ======================
fixtures_data = [
    {"date": "2026-04-11", "home_team": "West Ham United",              "away_team": "Wolverhampton Wanderers"},
    {"date": "2026-04-11", "home_team": "Arsenal",     "away_team": "Bournemouth"},
    {"date": "2026-04-11", "home_team": "Brentford",             "away_team": "Everton"},
    {"date": "2026-04-11", "home_team": "Burnley",   "away_team": "Brighton & Hove Albion"},
    {"date": "2026-04-12", "home_team": "Liverpool",   "away_team": "Fulham"},
    {"date": "2026-04-12", "home_team": "Crystal Palace",     "away_team": "Newcastle"},
    {"date": "2026-04-12", "home_team": "Nottingham Forest",   "away_team": "Aston Villa"},
    {"date": "2026-04-12", "home_team": "Sunderland",      "away_team": "Tottenham Hotspur"},
    {"date": "2026-04-12", "home_team": "Chelsea",         "away_team": "Manchester City"},
    {"date": "2026-04-14", "home_team": "Manchester United", "away_team": "Leeds United"}
]  # ← keep the same 10 fixtures you had before

upcoming = pd.DataFrame(fixtures_data)
upcoming['date'] = pd.to_datetime(upcoming['date'])

if validate_fixtures(upcoming, df_elo):
    predictions = predict_upcoming(upcoming, df_elo)

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("⚙️ Controls")
        st.button("🔄 Refresh Predictions", type="primary")
        st.caption("Matchweek April 11–14 2026")

    # ====================== MAIN TABLE ======================
    st.subheader(f"Matchweek Predictions – {predictions['date'].dt.date.min()} to {predictions['date'].dt.date.max()}")


    # Style the table
    def style_table(df):
        def color_prob(val):
            if isinstance(val, str) and '%' in val:
                p = float(val.strip('%')) / 100
                if p > 0.60: return 'background-color: #006400; color: white;'
                if p > 0.45: return 'background-color: #1E90FF; color: white;'
                return 'background-color: #8B0000; color: white;'
            return ''

        styled = df.style.applymap(color_prob, subset=['prob_home', 'prob_draw', 'prob_away'])
        return styled


    display_cols = ['date', 'home_team', 'away_team',
                    'prob_home', 'prob_draw', 'prob_away',
                    'fair_odds_home', 'fair_odds_draw', 'fair_odds_away']

    styled_df = predictions[display_cols].copy()
    styled_df['prob_home'] = styled_df['prob_home'].apply(lambda x: f"{x:.1%}")
    styled_df['prob_draw'] = styled_df['prob_draw'].apply(lambda x: f"{x:.1%}")
    styled_df['prob_away'] = styled_df['prob_away'].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        style_table(styled_df),
        use_container_width=True,
        hide_index=True
    )

    # Download
    csv = predictions.to_csv(index=False)
    st.download_button("📥 Download full predictions as CSV", csv, "matchweek_predictions.csv", "text/csv")

st.caption("✅ Team names strictly validated • Model trained on 4,489 historical matches")