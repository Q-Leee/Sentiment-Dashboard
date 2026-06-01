from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SAMPLE_CSV = DATA_DIR / "sample_reviews.csv"
AMAZON_CSV = PROJECT_ROOT / "Amazon_Reviews.csv"
if not AMAZON_CSV.exists():
    AMAZON_CSV = DATA_DIR / "Amazon_Reviews.csv"
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
