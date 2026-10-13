"""Single-URL prediction: from predict import predict."""
import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Make src/ importable from the repo root.

from model import Scorer
from preprocess import clean_url

INVALID = {"label": "clean", "probability": 0.5}  # Neutral fallback for empty or unusable inputs.


@lru_cache(maxsize=1)
def _model() -> Scorer:
    return Scorer("advanced")  # loaded on first call, not on import


def predict(url) -> dict:
    """Return {"label": "phishing" | "clean", "probability": float in [0, 1]}.
    Empty or unusable input returns a neutral {"label": "clean", "probability": 0.5}."""
    try:
        if not clean_url(url):
            return dict(INVALID)
        model = _model()
        probability = float(model.predict_proba([url])[0])
        if not math.isfinite(probability):
            return dict(INVALID)
    except Exception:  # Feature extraction should not break callers.
        return dict(INVALID)
    # Use the same validation-selected cutoff as batch scoring.
    return {"label": "phishing" if probability >= model.threshold else "clean", "probability": probability}
