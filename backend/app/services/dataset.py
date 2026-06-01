from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.config import AMAZON_CSV, DATA_DIR, PROJECT_ROOT, SAMPLE_CSV
from app.services.preprocess import clean_text

_RATING_RE = re.compile(r"rated\s+(\d)\s+out\s+of\s+5", re.IGNORECASE)

_TEXT_COLUMNS = ("text", "review", "comment", "content", "body", "review_text")
_LABEL_COLUMNS = ("label", "sentiment", "polarity", "class")
_RATING_COLUMNS = ("rating", "stars", "score")

_LABEL_MAP = {
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "good": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "2": "neutral",
    "3": "neutral",
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "bad": "negative",
}


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def _stars_to_label(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        m = _RATING_RE.search(value)
        if m:
            stars = int(m.group(1))
        else:
            try:
                stars = int(float(value.strip()))
            except ValueError:
                return None
    else:
        try:
            stars = int(float(value))
        except (TypeError, ValueError):
            return None
    if stars <= 2:
        return "negative"
    if stars >= 4:
        return "positive"
    if stars == 3:
        return "neutral"
    return None


def _normalize_label(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return _LABEL_MAP.get(key)


def _rating_to_label(rating: str) -> str | None:
    return _stars_to_label(rating)


def load_amazon_reviews(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Review Text" not in raw.columns:
        raise ValueError(f"Amazon CSV must have 'Review Text' column: {path}")

    title = raw["Review Title"].fillna("").astype(str) if "Review Title" in raw.columns else ""
    body = raw["Review Text"].fillna("").astype(str)
    if isinstance(title, pd.Series):
        text = (title.str.strip() + " " + body.str.strip()).str.strip()
    else:
        text = body.str.strip()

    labels = raw["Rating"].map(_rating_to_label) if "Rating" in raw.columns else None
    return _finalize_frame(pd.DataFrame({"text": text, "label": labels}), min_len=10)


def _finalize_frame(df: pd.DataFrame, min_len: int = 1) -> pd.DataFrame:
    df["text"] = df["text"].map(clean_text)
    df = df.dropna(subset=["label"])
    df = df[df["text"].str.len() >= min_len]
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(("positive", "neutral", "negative"))]
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_labeled_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    cols = list(raw.columns)

    if "Review Text" in cols:
        return load_amazon_reviews(path)

    text_col = _pick_column(cols, _TEXT_COLUMNS)
    if not text_col:
        raise ValueError(f"No text column in {path.name}. Use one of: {_TEXT_COLUMNS}")

    label_col = _pick_column(cols, _LABEL_COLUMNS)
    rating_col = _pick_column(cols, _RATING_COLUMNS)

    text = raw[text_col].fillna("").astype(str).str.strip()
    if label_col:
        labels = raw[label_col].map(_normalize_label)
    elif rating_col:
        labels = raw[rating_col].map(_stars_to_label)
    else:
        raise ValueError(
            f"No label/rating column in {path.name}. "
            f"Use label/sentiment or numeric rating (1-5)."
        )

    return _finalize_frame(pd.DataFrame({"text": text, "label": labels}))


def _is_amazon_file(path: Path) -> bool:
    return path.name.lower().startswith("amazon") or path.resolve() == AMAZON_CSV.resolve()


def discover_training_files() -> list[Path]:
    """Collect CSVs: Amazon (if any) + backend/data/*.csv + other *.csv in project root."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if not path.exists() or resolved in seen or path.suffix.lower() != ".csv":
            return
        seen.add(resolved)
        ordered.append(path)

    if AMAZON_CSV.exists():
        add(AMAZON_CSV)

    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.csv")):
            add(path)

    for path in sorted(PROJECT_ROOT.glob("*.csv")):
        if _is_amazon_file(path):
            continue
        add(path)

    if not ordered and SAMPLE_CSV.exists():
        add(SAMPLE_CSV)

    return ordered


def load_single_file(path: Path) -> pd.DataFrame:
    if _is_amazon_file(path) or path.name.lower().startswith("amazon"):
        return load_amazon_reviews(path)
    return load_labeled_csv(path)


def load_training_dataframe(csv_path: Path | None = None) -> tuple[pd.DataFrame, str, list[dict]]:
    if csv_path is not None:
        df = load_single_file(csv_path)
        return df, csv_path.name, [{"file": csv_path.name, "rows": len(df)}]

    files = discover_training_files()
    if not files:
        raise FileNotFoundError("No training CSV found. Add Amazon_Reviews.csv or files under backend/data/.")

    frames: list[pd.DataFrame] = []
    sources: list[dict] = []

    for path in files:
        try:
            part = load_single_file(path)
        except Exception as exc:
            sources.append({"file": path.name, "rows": 0, "error": str(exc)})
            continue
        if part.empty:
            sources.append({"file": path.name, "rows": 0, "skipped": "empty"})
            continue
        frames.append(part)
        sources.append({"file": path.name, "rows": len(part)})

    if not frames:
        raise ValueError("All training files failed or were empty.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)

    names = ", ".join(s["file"] for s in sources if s.get("rows"))
    return merged, names, sources


def resolve_training_csv() -> Path:
    files = discover_training_files()
    if files:
        return files[0]
    return SAMPLE_CSV
