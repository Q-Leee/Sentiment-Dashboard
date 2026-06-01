from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import MODEL_PATH
from app.services.dataset import load_training_dataframe
from app.services.preprocess import augment_no_apostrophe_variants, clean_text, rule_label


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                    preprocessor=clean_text,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_and_save(csv_path: Path | None = None, model_path: Path | None = None) -> dict:
    model_path = model_path or MODEL_PATH

    df, source, file_stats = load_training_dataframe(csv_path)
    label_counts = df["label"].value_counts().to_dict()

    # Teach model both "don't" and "dont" spellings
    aug_rows: list[dict] = []
    for text, label in zip(df["text"], df["label"]):
        for variant in augment_no_apostrophe_variants(text):
            aug_rows.append({"text": variant, "label": label})
    train_df = pd.DataFrame(aug_rows).drop_duplicates(subset=["text"])

    X = train_df["text"]
    y = train_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    accuracy = float(pipe.score(X_test, y_test))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    return {
        "samples": len(df),
        "training_rows_augmented": len(train_df),
        "accuracy": round(accuracy, 4),
        "model_path": str(model_path),
        "dataset": source,
        "sources": file_stats,
        "labels": {str(k): int(v) for k, v in label_counts.items()},
    }


def load_pipeline() -> Pipeline:
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


def predict_texts(texts: list[str]) -> list[dict]:
    pipe = load_pipeline()
    pairs = [(raw, clean_text(raw)) for raw in texts if raw and raw.strip()]
    if not pairs:
        return []

    raw_texts = [p[0] for p in pairs]
    normalized = [p[1] for p in pairs]

    # preprocessor=clean_text inside vectorizer; pass raw so behavior matches training source text
    labels = pipe.predict(raw_texts)
    try:
        proba = pipe.predict_proba(raw_texts)
        classes = list(pipe.classes_)
    except Exception:
        proba = None
        classes = []

    results: list[dict] = []
    for i, (raw, norm) in enumerate(zip(raw_texts, normalized)):
        ml_label = str(labels[i])
        ml_confidence = None
        if proba is not None and classes:
            idx = classes.index(ml_label) if ml_label in classes else 0
            ml_confidence = round(float(proba[i][idx]), 4)

        rule = rule_label(norm)
        if rule:
            final_label = rule
            source = "rule"
            # Show high certainty when a clear phrase rule matched
            if proba is not None and classes and rule in classes:
                ridx = classes.index(rule)
                model_p = float(proba[i][ridx])
                confidence = round(max(model_p, 0.92), 4)
            else:
                confidence = 0.92
        else:
            final_label = ml_label
            source = "model"
            confidence = ml_confidence

        results.append(
            {
                "text": raw.strip(),
                "text_normalized": norm,
                "label": final_label,
                "confidence": confidence,
                "model_confidence": ml_confidence,
                "source": source,
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    positive = sum(1 for r in results if r["label"] == "positive")
    neutral = sum(1 for r in results if r["label"] == "neutral")
    negative = sum(1 for r in results if r["label"] == "negative")
    other = total - positive - neutral - negative
    return {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "other": other,
        "positive_pct": round(100 * positive / total, 1) if total else 0,
        "neutral_pct": round(100 * neutral / total, 1) if total else 0,
        "negative_pct": round(100 * negative / total, 1) if total else 0,
    }
