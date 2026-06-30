"""
Phishing Email Detection — Gradio Web App
Loads the trained model and TF-IDF vectorizer, then exposes a simple
interface where a user can paste email text and receive a prediction.

Run locally:
    python app.py

Deploy:
    Push this file, requirements.txt, and the models/ folder to a
    Hugging Face Space with the Gradio SDK selected.
"""

import re
import string
import joblib
import gradio as gr


# ---------------------------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------------------------
model = joblib.load("models/phishing_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


def clean_text(text):
    """Match the same preprocessing used during training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_email(email_text):
    if not email_text or not email_text.strip():
        return "Please paste an email to analyze.", ""

    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence = probabilities[prediction] * 100

    result = f"{label}"
    detail = f"Confidence: {confidence:.1f}%"

    return result, detail


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
example_phishing = (
    "Dear Customer, your account has been suspended due to suspicious activity. "
    "Click here immediately to verify your identity and restore access within 24 hours "
    "or your account will be permanently closed."
)

example_legitimate = (
    "Hi team, just a reminder that our weekly sync is moved to 3 PM tomorrow. "
    "Please review the attached agenda before the meeting. Thanks!"
)

demo = gr.Interface(
    fn=predict_email,
    inputs=gr.Textbox(
        lines=8,
        label="Email text",
        placeholder="Paste the full email content here...",
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence score"),
    ],
    title="Phishing Email Detector",
    description=(
        "Paste any email text below to check whether it is likely phishing or "
        "legitimate. This model uses TF-IDF text vectorization with a "
        "supervised classifier, achieving 97.2% accuracy on a held-out test set."
    ),
    examples=[
        [example_phishing],
        [example_legitimate],
    ],
)

if __name__ == "__main__":
    demo.launch()
