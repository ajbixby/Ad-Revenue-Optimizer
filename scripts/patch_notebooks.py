"""Patch workshop notebooks for offline local data mode."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = [
    Path("notebooks/student_workbook.ipynb"),
    Path("student_workbook.ipynb"),
]

CELL1 = """\
import sys
from pathlib import Path

# Project root (works from repo root or notebooks/)
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_SOURCE = "local"  # "local" (offline) or "live" (workshop server)
SERVER = "https://lse-mayday.onrender.com"
TEAM_NAME = "The_Meat_Team"  # Pick a unique team name

if sys.version_info[:2] != (3, 12):
    print(f'WARNING: workshop server expects Python 3.12 — you are on '
          f'{sys.version_info.major}.{sys.version_info.minor}. Model upload '
          f'may fail with pickle errors.')
"""

CELL3 = """\
from src.notebook_query import query, ensure_data, get_schema, print_schema

ensure_data()  # generates synthetic parquet tables on first run
"""

CELL5 = """\
if DATA_SOURCE == "live":
    import requests
    resp = requests.post(
        f"{SERVER}/api/teams/{TEAM_NAME}/register",
        json={"members": ["AJ", "Anna", "Lynlee", "Eren"]},
    )
    print(resp.json())
else:
    print(f"Offline mode — skipping team registration.")
    print(f"Team name for model export: {TEAM_NAME}")
"""

CELL6_MD = """\
---

# Phase 1: Explore the Data (45 min)

You have access to an e-commerce ad database with four tables:
- **users** — 10,000 users with demographics and behavior
- **ads** — 200 ad creatives with product and creative metadata
- **impressions** — 500,000 ad impressions (did the user click?)
- **conversions** — purchases that followed a click

> **Offline mode:** SQL queries run locally against synthetic parquet data via DuckDB.
> Set `DATA_SOURCE = "live"` in cell 1 if the workshop server is available.

## 1.1 Get the lay of the land"""

CELL7 = """\
# Check what we're working with
print_schema(get_schema())"""

PULL_CELL = """\
from src.data import pull_training_data

# Load joined training data from local parquet (or live API if DATA_SOURCE == "live")
print("Loading training data...")
df = pull_training_data(max_rows=100_000, source=DATA_SOURCE)
print(f"\\nTotal: {len(df):,} rows")
df.head()"""

UPLOAD_CELL = """\
from pathlib import Path

# Upload to the server (live mode only)
if DATA_SOURCE == "live":
    with open(model_path, 'rb') as f:
        resp = requests.post(
            f"{SERVER}/api/teams/{TEAM_NAME}/model",
            files={"model": (model_path, f, "application/octet-stream")}
        )

    result = resp.json()
    print(f"Upload result: {result}")
    if resp.ok:
        print(f"\\nModel uploaded successfully!")
        print(f"  Type: {result.get('model_type')}")
        print(f"  Validation prediction: {result.get('validation_prediction'):.4f}")
        print(f"  Validation latency: {result.get('validation_latency_ms'):.2f} ms")
    else:
        print(f"\\nERROR: {result}")
else:
    print("Offline mode — model saved locally. Run `streamlit run app/streamlit_app.py` to demo scoring.")
"""

DASHBOARD_MD = """\
## 4.2 Watch the live dashboard

> **Offline mode:** The workshop server is unavailable. Use the Streamlit demo instead:
> `streamlit run app/streamlit_app.py`

When the server is live, open the dashboard to watch the A/B test in real time.

**Dashboard URL**: `{SERVER}/dashboard`"""

LEADERBOARD_CELL = """\
if DATA_SOURCE == "live":
    resp = requests.get(f"{SERVER}/api/leaderboard")
    print(resp.json())
else:
    print("Offline mode — no live leaderboard. Compare models offline using revenue lift in the evaluation cells.")
"""


def set_cell_source(cells: list, idx: int, source: str) -> None:
    cells[idx]["source"] = [line + "\n" for line in source.split("\n")]
    if cells[idx]["source"]:
        cells[idx]["source"][-1] = cells[idx]["source"][-1].rstrip("\n")


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    cells = nb["cells"]

    set_cell_source(cells, 1, CELL1)
    set_cell_source(cells, 3, CELL3)
    set_cell_source(cells, 5, CELL5)
    set_cell_source(cells, 6, CELL6_MD)
    set_cell_source(cells, 7, CELL7)

    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if "def pull_training_data" in src and "pull_training_data()" in src:
            set_cell_source(cells, i, PULL_CELL)
        if "# Upload to the server" in src and "requests.post" in src and "model" in src:
            set_cell_source(cells, i, UPLOAD_CELL)
        if "Watch the live dashboard" in src and cell["cell_type"] == "markdown":
            set_cell_source(cells, i, DASHBOARD_MD)
        if "api/leaderboard" in src:
            set_cell_source(cells, i, LEADERBOARD_CELL)

    path.write_text(json.dumps(nb, indent=1))
    print(f"Patched {path}")


if __name__ == "__main__":
    for nb_path in NOTEBOOKS:
        if nb_path.exists():
            patch_notebook(nb_path)
