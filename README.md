# Sentiment Dashboard

Classify reviews and comments as **positive**, **neutral**, or **negative**, then visualize counts and ratios in a simple dashboard.

Personal ML portfolio project: offline training → FastAPI serving → React UI.

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, scikit-learn (TF-IDF + Logistic Regression) |
| Frontend | React, Vite, Recharts |
| Inference | Hybrid **ML model + phrase rules** (negation, `not bad`, etc.) |

## Architecture

```text
[React UI :5173]  --/api-->  [FastAPI :8011]
                                  |
                    joblib pipeline + rule overrides
                                  |
                    CSV merge (Amazon + backend/data/*.csv)
```

## Quick start (Windows)

### 1) Backend

```powershell
cd c:\Users\gudrb\Desktop\projects\sentiment-dashboard\backend
.\run.ps1
```

First run: creates `.venv`, installs deps, trains the model, starts the API on **8011**.

Health: http://127.0.0.1:8011/health  
API docs: http://127.0.0.1:8011/docs

### 2) Frontend (new terminal)

```powershell
cd c:\Users\gudrb\Desktop\projects\sentiment-dashboard\frontend
npm install
npm run dev
```

UI: http://127.0.0.1:5173  
Vite proxies `/api` → `http://127.0.0.1:8011`.

### Manual commands

**Train only**

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\train.py
```

**Server only**

```powershell
cd backend
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8011
```

## Training data

All matching CSVs are merged on train (deduped by `text`).

| Location | Examples |
|----------|----------|
| Project root | `Amazon_Reviews.csv` |
| `backend/data/` | `sample_reviews.csv`, custom files |

### Amazon format

Columns: `Review Title`, `Review Text`, `Rating` (`Rated N out of 5 stars`)

| Stars | Label |
|-------|-------|
| 1–2 | negative |
| 3 | neutral |
| 4–5 | positive |

### Generic format

`text` + `label`, or `review` + `sentiment`, or `text` + numeric `rating` (1–5).

Labels: `positive` / `neutral` / `negative` (also `pos`, `neg`, `neu`, `0`, `1`, `2`, `3`).

> **Note:** `Amazon_Reviews.csv` is large (~12MB). Keep it local or add to `.gitignore`; clone the repo and place the file in the project root before training.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + training file list |
| POST | `/train` | Retrain from all discovered CSVs |
| POST | `/analyze` | `{ "texts": ["...", "..."] }` (max 500) |
| POST | `/analyze/csv` | Upload CSV (auto-detect text column) |

**Analyze response (per row)**

| Field | Meaning |
|-------|---------|
| `label` | Final class: `positive` / `neutral` / `negative` |
| `confidence` | How sure the classifier is (not sentiment intensity) |
| `source` | `model` or `rule` (phrase override) |
| `model_confidence` | ML probability before rules |

## Hybrid inference

1. **ML** — TF-IDF + logistic regression on merged training data (~21k rows with Amazon CSV).
2. **Rules** — Clear phrases the bag-of-words model often misses, e.g.:
   - `not good` → negative
   - `not bad`, `not too bad` → neutral
   - `won't come back`, `don't want` → negative

Rules run after ML and set `source: "rule"` when matched.

## Project layout

```text
sentiment-dashboard/
  Amazon_Reviews.csv          # optional, local (not in git by default)
  backend/
    app/
      main.py                 # FastAPI routes
      services/
        dataset.py            # CSV load + merge
        preprocess.py         # clean_text + phrase rules
        sentiment.py          # train + predict
    data/sample_reviews.csv
    models/                   # sentiment_pipeline.joblib (generated)
    scripts/train.py
    run.ps1
  frontend/
    src/App.tsx               # dashboard UI
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `WinError 10013` on start | Port **8011** already in use — stop other uvicorn/python on that port, or change port in `run.ps1` + `frontend/vite.config.ts` |
| Old/wrong predictions | Restart backend after code changes; re-run `train.py` after CSV updates |
| `python` not found | Use `py` or `.\.venv\Scripts\python.exe` |

## Optional next steps

- DistilBERT as a second model (`model=classic\|bert`) for context-heavy sentences
- Drift monitoring when running as a long-lived service with live feedback
- Deploy: static frontend build + containerized FastAPI
