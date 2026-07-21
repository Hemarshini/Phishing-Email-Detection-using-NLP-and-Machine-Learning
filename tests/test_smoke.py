"""
Smoke tests for the Phishing Email Detector.

These are intentionally lightweight "does it actually work end-to-end" checks,
not exhaustive unit tests — the goal is to catch a broken build (missing model
file, incompatible library version, model that crashes on real input) before
it ever reaches deployment, which is exactly what CI should guard against.
"""
import os
import joblib
import pytest

MODEL_PATH = "models/phishing_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"


def test_model_artifacts_exist():
    """The trained model and vectorizer must be present and committed to the repo."""
    assert os.path.exists(MODEL_PATH), f"Missing model file at {MODEL_PATH}"
    assert os.path.exists(VECTORIZER_PATH), f"Missing vectorizer file at {VECTORIZER_PATH}"


def test_model_and_vectorizer_load_without_error():
    """The saved artifacts must actually deserialize correctly (catches version-mismatch bugs)."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    assert model is not None
    assert vectorizer is not None


@pytest.mark.parametrize("sample_email,expected_signal", [
    (
        "Dear customer, your account has been suspended. Click here immediately "
        "to verify your identity or your account will be permanently deleted.",
        "phishing-like",
    ),
    (
        "Hi team, attaching the quarterly report for your review ahead of "
        "tomorrow's meeting. Let me know if you have any questions.",
        "legitimate-like",
    ),
])
def test_model_predicts_on_realistic_samples(sample_email, expected_signal):
    """
    Runs the full inference path on two realistic examples — one written in
    classic urgency-driven phishing style, one written as an ordinary work email.
    This doesn't assert a specific label (that would make the test brittle to
    retraining), it just confirms the pipeline runs end-to-end without error
    and returns a valid prediction shape.
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    features = vectorizer.transform([sample_email])
    prediction = model.predict(features)

    assert len(prediction) == 1
    assert prediction[0] is not None


def test_url_heuristics_module_importable():
    """Confirms the rule-based URL heuristic layer at least imports cleanly."""
    import url_heuristics  # noqa: F401
