# World Cup ML Predictor

XGBoost 3-class model (Home / Draw / Away) trained on international tournaments with
StatBomb open-data xG features, mirroring the Augo Premier League pipeline.

## One-time setup / retrain

```bash
python3 world_cup/data_pipeline.py
python3 world_cup/feature_engineering.py
python3 world_cup/train_xgb.py
```

## Before each round

1. Update `world_cup/fixtures.csv` (`round`, `date`, `home_team`, `away_team`).
2. Run inference:

```bash
python3 world_cup/run_pipeline.py --round 1
```

Outputs:

- `world_cup/predictions_cache.json`
- `world_cup/predictions_history/R{n}.json`

## Data sources

- **Primary**: [StatBomb open data](https://github.com/statsbomb/open-data) (WC 2018/2022, Euro 2020/2024, Copa 2024, AFCON 2023)
- **Optional**: `FOOTBALL_DATA_API_KEY` in `.env` for football-data.org supplements
- **Optional**: `ODDS_API_KEY` (used by Augo UI when wired)

## Features (20)

Rolling 10-game averages of xG, xGA, goals, shots on target, form, plus ELO diff,
head-to-head stats, and knockout-stage flag.

## Grading

Add completed scores to `world_cup/results.csv` (same schema as Premier League `results.csv`).
The Augo UI World Cup mode reads this file for history grading.
