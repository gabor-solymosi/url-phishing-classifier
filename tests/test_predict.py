import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]  # Resolve imports consistently from any pytest cwd.
sys.path.insert(0, str(ROOT))

import predict
import explain as explanation_module
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


def test_plain_predict_has_only_label_and_probability():
    # Explanations are opt-in so existing callers keep the same result shape.
    assert set(predict.predict("https://example.com/")) == {"label", "probability"}


def test_predict_explanation_uses_template_and_keeps_score(monkeypatch):
    url = "https://www.PayPal-secure.login.tk/webscr?cmd=x"
    monkeypatch.setattr(explanation_module, "_load_llm", lambda: None)
    plain = predict.predict(url)
    explained = predict.predict(url, explain=True)
    assert explained["label"] == plain["label"]
    assert explained["probability"] == plain["probability"]
    assert explained["explanation"]["text"]
    assert explained["explanation"]["facts"] == explained["explanation"]["text"]
    assert explained["explanation"]["generated_by"] == "template"


def test_explain_bad_inputs_do_not_raise():
    for url in ("", "%%%://"):
        explanation_module.explain(url, use_llm=False)


def test_explanation_group_values_and_facts_are_human_readable():
    url = "https://www.PayPal-secure.login.tk/webscr?cmd=x"
    features = explanation_module.advanced_features([url]).toarray()
    cleaned = "https://www.paypal-secure.login.tk/webscr?cmd=x"
    assert explanation_module._value("host_patterns", features, cleaned) == (
        "www.paypal-secure.login.tk"
    )
    assert explanation_module._value("path_patterns", features, cleaned) == (
        "/webscr?cmd=x"
    )
    assert explanation_module._value("url_words", features, cleaned) == (
        "https www paypal secure"
    )
    result = explanation_module.explain(url, use_llm=False)
    assert result["facts"] == result["text"]
    assert result["facts"].startswith("This URL is flagged as phishing")
    assert "The strongest signal is" in result["facts"]


def test_paypal_narrative_contains_first_reason_phrase():
    url = "https://www.PayPal-secure.login.tk/webscr?cmd=x"
    cleaned = explanation_module.clean_url(url)
    features = explanation_module.advanced_features([cleaned]).toarray()
    result = explanation_module.explain(url, use_llm=False)
    reason_key = next(
        key for key, name in explanation_module.NAMES.items()
        if name == result["reasons"][0]["name"]
    )
    phrase = explanation_module._phrase(reason_key, features, cleaned)
    assert result["facts"].startswith("This URL is flagged as phishing")
    assert phrase in result["facts"]


def test_narrative_phrases_filter_tokens_and_pluralize_counts():
    url = "https://www.paypal-secure.login.tk/webscr?cmd=x"
    features = explanation_module.advanced_features([url]).toarray()
    assert explanation_module._phrase("url_words", features, url) == (
        "the words in the URL (paypal secure login tk)"
    )
    assert explanation_module._phrase("log_suspicious_words", features, url) == (
        "the account-related words 'login', 'secure' and 'webscr'"
    )

    empty_url = "https://example.com"
    empty_features = explanation_module.advanced_features([empty_url]).toarray()
    assert explanation_module._phrase(
        "log_path_slashes", empty_features, empty_url
    ) == "no slashes in the path"
    assert explanation_module._phrase(
        "log_len_path", empty_features, empty_url
    ) == "an empty path"


def test_llm_guard_requires_first_two_words_of_reason():
    prompt = "Verdict: phishing (score 0.98)"
    assert not explanation_module._valid_text(
        "The score indicates a phishing URL.", prompt, "phishing", "suspicious words"
    )
    assert explanation_module._valid_text(
        "Suspicious words indicate a phishing URL.", prompt, "phishing", "suspicious words"
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
