# Sentiment Dashboard

Classify reviews and comments as **positive** or **negative**, then visualize counts and ratios in a simple dashboard.

## Stack

- **Backend:** FastAPI + scikit-learn (TF-IDF + Logistic Regression)
- **Frontend:** React + Vite + Recharts
- **Training data:** Multiple CSVs are merged automatically (duplicates removed by text)

## Run (Windows)

### 1) Backend

```powershell
cd c:\Users\gudrb\Desktop\projects\sentiment-dashboard\backend
.\run.ps1
```

On first run: creates venv, installs dependencies, trains the model, then serves the API on port **8010**.

Health check: http://127.0.0.1:8010/health

### 2) Frontend (new terminal)

```powershell
cd c:\Users\gudrb\Desktop\projects\sentiment-dashboard\frontend
npm install
npm run dev
```

UI: http://127.0.0.1:5173  
Vite proxies `/api` to the backend.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/train` | Retrain from sample CSV |
| POST | `/analyze` | JSON `{ "texts": ["...", "..."] }` |
| POST | `/analyze/csv` | Upload CSV (`text` column or first column) |

## Training data (multiple sources OK)

All matching CSVs are combined on train:

| Location | Examples |
|----------|----------|
| Project root | `Amazon_Reviews.csv` (Amazon export) |
| `backend/data/*.csv` | `sample_reviews.csv`, `yelp.csv`, … |

**Amazon format:** `Review Title`, `Review Text`, `Rating` (`Rated N out of 5 stars`) → 1–2★ negative, 4–5★ positive, 3★ skipped.

**Generic format:** columns like `text` + `label`, or `review` + `sentiment`, or `text` + numeric `rating` (1–5). Labels: `positive` / `negative` (also `pos`, `neg`, `0`, `1`).

Retrain:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\train.py
```

Or call `POST http://127.0.0.1:8010/train`

## Optional next steps

- Add a `neutral` label
- Swap in BERT / DistilBERT for better accuracy
- Connect a real review export (Amazon, app store, etc.)
