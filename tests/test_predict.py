import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]  # Resolve imports consistently from any pytest cwd.
sys.path.insert(0, str(ROOT))

import predict
from model import Scorer
from preprocess import STRUCTURE_NAMES, structure


@pytest.fixture(scope="module")
def scorer():
    # Reuse one ONNX session across the edge-case parameter set.
    return Scorer("advanced")


@pytest.mark.parametrize(
    "url",
    # Mix bad types and malformed strings with real-world URL edge cases.
    [
        "http://paypal-login.example.com/verify",
        "",
        "   ",
        "\x00\x01\n",
        None,
        12345,
        3.5,
        b"http://x.com",
        ["a"],
        "not a url",
        "http://[::1",
        "://",
        "https://user:pw@1.2.3.4:8080/a?b=c#d",
        "http://例え.テスト/パス",
        "a" * 100000,
        "\ud800",
        "javascript:alert(1)",
        "HTTPS://WWW.GOOGLE.COM/",
    ],
)
def test_predict_handles_odd_inputs(url, scorer):
    result = predict.predict(url)
    assert set(result) == {"label", "probability"}
    assert result["label"] in {"phishing", "clean"}
    assert type(result["probability"]) is float
    assert math.isfinite(result["probability"])
    assert 0 <= result["probability"] <= 1
    # The public label must use the same cutoff as the batch scorer.
    assert (result["label"] == "phishing") == (
        result["probability"] >= scorer.threshold
    )


@pytest.mark.parametrize("url", ["", "   ", None])
def test_empty_inputs_are_neutral(url):
    # Empty values should not trigger model loading or parsing failures.
    assert predict.predict(url) == {"label": "clean", "probability": 0.5}


def test_prediction_is_case_insensitive():
    # Cleaning lowercases before feature extraction.
    assert predict.predict("https://www.google.com/") == predict.predict(
        "HTTPS://WWW.GOOGLE.COM/"
    )


def test_import_does_not_load_model():
    # A fresh process catches accidental eager ONNX loads.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import predict; "
            "assert predict._model.cache_info().currsize == 0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_structure_names_match_feature_count():
    # Named notebook columns must stay aligned with structure() output order.
    assert len(STRUCTURE_NAMES) == len(structure("http://a.com", "a.com", "/"))


def test_xgboost_scorer_returns_finite_probabilities():
    probabilities = Scorer("xgboost").predict_proba(
        ["https://example.com/", "not a url", "", "https://user:pw@127.0.0.1/a?b=c#d"]
    )
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
