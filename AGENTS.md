# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Augo is a Premier League match-outcome prediction app built with Python/Reflex. It uses a soft-voting ensemble ML model and serves predictions via a full-stack web UI (Reflex compiles to React/Next.js).

### Running the Application

```bash
reflex run
```

- Frontend: http://localhost:3000/
- Backend: http://localhost:8000/
- On first run after fresh install, `reflex init` is required before `reflex run`.

### Key Gotchas

- **No `python` binary**: The system only has `python3`. Use `python3` for all commands.
- **System `packaging` conflict**: When installing deps, you may need `pip install --ignore-installed packaging` before `pip install -r requirements.txt` due to a system-managed `packaging` package without a RECORD file.
- **Reflex manages its own Node.js/bun**: No need to install Node manually. Reflex downloads and manages its own frontend toolchain internally.
- **No database required**: All persistence is file-based (JSON/CSV). Reflex uses SQLite internally (auto-created `.db` file).
- **`predictions_cache.json` must exist**: The app reads this at boot. It ships in the repo, but if missing, run `python3 run_pipeline.py` to regenerate (requires model `.pkl` files to be present).
- **`ODDS_API_KEY` is optional**: Without it, `run_pipeline.py` skips live odds and the app still functions.

### Lint / Type Checking

```bash
ruff check .       # linting (pre-existing warnings exist, no blocking errors)
pyright .          # type checking (optional, may have pre-existing issues)
```

### Testing

No automated test suite exists in this repo. Validation is done by running the app and interacting with the UI.

### Build

The Reflex framework handles both frontend compilation and backend serving in one command (`reflex run`). There is no separate build step for development.
