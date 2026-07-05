# Predict, Ship, Compete

A live-A/B-test ad-targeting workshop. Build a model, upload it to the workshop server, and watch it compete against your classmates' models in real time. The team with the highest revenue per impression wins.

## Data (workshop server offline)

The LSE workshop server is no longer available. This repo includes a **synthetic ad funnel**
(`data/synthetic_impressions.parquet`) calibrated to the original workshop statistics.

```bash
# Regenerate data (optional)
python -m src.simulate --rows 500000

# Train locally (no upload)
python mayday_model.py --skip-upload

# First-time setup: auto-generates data if parquet is missing
python mayday_model.py --skip-upload
```

Use `--source live` to pull from the workshop API if it comes back online.

### Notebook (offline SQL + EDA)

```bash
jupyter notebook notebooks/student_workbook.ipynb
```

Cell 1 sets `DATA_SOURCE = "local"` (default). All `query(...)` calls run against local parquet via DuckDB. Set `DATA_SOURCE = "live"` to use the workshop server.

### Streamlit scoring demo

```bash
streamlit run app/streamlit_app.py
```

Interactive ad ranking: score a single ad or compare three presets for the same user profile.

## Setup (do this before the workshop — about 5 minutes)

You need **Python 3.12 specifically**. Models pickled under any other minor version (3.11, 3.13, ...) cannot be loaded by the server, and you'll see cryptic errors at the upload step. If you don't have 3.12, the easiest path is Miniconda:

1. Install Miniconda if you don't have it: <https://docs.conda.io/en/latest/miniconda.html>
2. Open a terminal and run:

   ```bash
   git clone https://github.com/LSE-Methodology/mayday-live.git
   cd mayday-live
   conda create -n mayday python=3.12 -y
   conda activate mayday
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook notebooks/student_workbook.ipynb
   ```

If you'd rather use a plain virtualenv: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (you'll need a system Python 3.12 already installed).

## During the workshop

1. Set `TEAM_NAME` in cell 1 (any unique alphanumeric name; underscores and hyphens OK)
2. Run the cells in order — they walk you through exploration, modelling, and uploading
3. When the instructor starts the live A/B test, watch the dashboard at <https://lse-mayday.onrender.com/dashboard>

The notebook itself contains all the guidance you need — including the SQL Explorer at <https://lse-mayday.onrender.com/sql> for ad-hoc queries.

## Common errors

- **`Model validation failed: lasti is not an int`** — your Python isn't 3.12. Recreate your environment with `conda create -n mayday python=3.12`.
- **`Model too large (max 50MB)`** — your pickled model is over the upload cap. Try a smaller ensemble or feature subset.
- **`Failed to unpickle model`** — most often the same Python-version issue, sometimes a missing custom class. Make sure the class your `predict()` lives in is defined in a notebook cell that runs before you save.
- **Slow data pull** — the server is shared by all 20 teams. Default pull is 100k rows; that's plenty to learn the patterns. The full 500k will be very slow when everyone hits at once.

## What you cannot see

The workshop server's source — the data-generating process, the user-segment profiles, the simulator — is intentionally not in this repo. Half the workshop is figuring out the patterns from the data.
