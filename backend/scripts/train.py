"""Train sentiment model from sample_reviews.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.sentiment import train_and_save  # noqa: E402

if __name__ == "__main__":
    info = train_and_save()
    print("Trained:", info)
