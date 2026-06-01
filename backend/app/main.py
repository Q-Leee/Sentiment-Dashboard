from __future__ import annotations

import io
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.sentiment import predict_texts, summarize, train_and_save

app = FastAPI(title="Sentiment Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class AnalyzeResponse(BaseModel):
    results: list[dict]
    summary: dict


@app.get("/health")
def health():
    from app.config import AMAZON_CSV, DATA_DIR, SAMPLE_CSV
    from app.services.dataset import discover_training_files

    files = discover_training_files()
    return {
        "status": "ok",
        "service": "sentiment-dashboard",
        "training_files": [p.name for p in files],
        "data_dir": str(DATA_DIR),
        "amazon_csv_present": AMAZON_CSV.exists(),
        "sample_csv_present": SAMPLE_CSV.exists(),
    }


@app.post("/train")
def train():
    info = train_and_save()
    return {"status": "trained", **info}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    results = predict_texts(body.texts)
    if not results:
        raise HTTPException(status_code=400, detail="No valid text to analyze")
    return AnalyzeResponse(results=results, summary=summarize(results))


@app.post("/analyze/csv", response_model=AnalyzeResponse)
async def analyze_csv(
    file: Annotated[UploadFile, File(...)],
    text_column: str = "text",
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}") from e

    col = text_column if text_column in df.columns else None
    if col is None:
        for candidate in ("text", "review", "comment", "content", "body"):
            if candidate in df.columns:
                col = candidate
                break
    if col is None:
        col = df.columns[0]

    texts = df[col].astype(str).tolist()
    results = predict_texts(texts)
    if not results:
        raise HTTPException(status_code=400, detail="No valid rows in CSV")
    return AnalyzeResponse(results=results, summary=summarize(results))
