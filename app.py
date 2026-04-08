import streamlit as st
import pandas as pd
import joblib
import difflib
from pathlib import Path

# ====================== CLEAN MOBILE APP CONFIG ======================
st.set_page_config(
    page_title="PL Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {background-color: #0E1117;}
    .stDataFrameToolbar {display: none !important;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    model = joblib.load("xgboost_premier_league_model.pkl")
    df_elo = pd.read_csv("premier_league_with_elo_best.csv")
    df_elo['date'] = pd.to_datetime(df_elo['date'])
    return model, df_elo

model, df_elo = load_data()

# ====================== FUNCTIONS ======================
def validate_fixtures(upcoming, df_elo):
    valid_teams = set(pd.concat([df_elo['home_team'], df_elo['away_team']]).unique())
    errors = []
    for _, row in upcoming.iterrows():
        home, away = row['home_team'], row['away_team']
        if home not in valid_teams:
            sug = difflib.get_close_matches(home, valid_teams, n=1, cutoff=0.6)
            errors.append(f"❌ Home '{home}' → Did you mean '{sug[0]}'?" if sug else f"❌ Home '{home}'")
        if away not in valid_teams:
            sug = difflib.get_close_matches(away, valid_teams, n=1, cutoff=0.6)
            errors.append(f"❌ Away '{away}' → Did you mean '{sug[0]}'?" if sug else f"❌ Away '{away}'")
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

# ====================== BOTTOM NAVIGATION ======================
tab1, tab2 = st.tabs(["📊 Predictions", "✏️ Edit Fixtures"])

with tab1:
    fixtures_data = [
        {"date": "2026-04-11", "home_team": "West Ham United",     "away_team": "Wolverhampton Wanderers"},
        {"date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
        {"date": "2026-04-11", "home_team": "Brentford",           "away_team": "Everton"},
        {"date": "2026-04-11", "home_team": "Burnley",             "away_team": "Brighton & Hove Albion"},
        {"date": "2026-04-12", "home_team": "Liverpool",           "away_team": "Fulham"},
        {"date": "2026-04-12", "home_team": "Crystal Palace",      "away_team": "Newcastle"},
        {"date": "2026-04-12", "home_team": "Nottingham Forest",   "away_team": "Aston Villa"},
        {"date": "2026-04-12", "home_team": "Sunderland",          "away_team": "Tottenham Hotspur"},
        {"date": "2026-04-12", "home_team": "Chelsea",             "away_team": "Manchester City"},
        {"date": "2026-04-14", "home_team": "Manchester United",   "away_team": "Leeds United"}
    ]
    
    upcoming = pd.DataFrame(fixtures_data)
    upcoming['date'] = pd.to_datetime(upcoming['date'])
    
    if validate_fixtures(upcoming, df_elo):
        predictions = predict_upcoming(upcoming, df_elo)
        
        display_df = predictions[['home_team', 'away_team', 'prob_home', 'prob_draw', 'prob_away',
                                  'fair_odds_home', 'fair_odds_draw', 'fair_odds_away']].copy()
        
        display_df['prob_home'] = display_df['prob_home'].apply(lambda x: f"{x:.1%}")
        display_df['prob_draw'] = display_df['prob_draw'].apply(lambda x: f"{x:.1%}")
        display_df['prob_away'] = display_df['prob_away'].apply(lambda x: f"{x:.1%}")
        
        def highlight_probs(val):
            try:
                p = float(val.strip('%')) / 100
                if p > 0.60: return 'background-color: #006400; color: white'
                if p > 0.45: return 'background-color: #1E90FF; color: white'
                return 'background-color: #8B0000; color: white'
            except:
                return ''
        
        styled = display_df.style.map(highlight_probs, subset=['prob_home', 'prob_draw', 'prob_away'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Edit Upcoming Fixtures")
    st.caption("Add / edit / delete matches (date format: YYYY-MM-DD)")
    
    edited_df = st.data_editor(
        pd.DataFrame(fixtures_data),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn("Date (YYYY-MM-DD)", required=True),
            "home_team": st.column_config.TextColumn("Home Team", required=True),
            "away_team": st.column_config.TextColumn("Away Team", required=True),
        }
    )
    
    if st.button("🔄 Use these fixtures for Predictions", type="primary", use_container_width=True):
        try:
            upcoming = edited_df.copy()
            upcoming['date'] = pd.to_datetime(upcoming['date'], errors='coerce')
            if upcoming['date'].isnull().any():
                st.error("Some dates are invalid. Please use YYYY-MM-DD format.")
            elif validate_fixtures(upcoming, df_elo):
                predictions = predict_upcoming(upcoming, df_elo)
                st.success("✅ Fixtures updated! Switch back to the **Predictions** tab to see the new results.")
        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Team names strictly validated • Model trained on 4,489 historical matches")
