"""
Phishing Email Detection — Gradio Web App
Loads the trained model and TF-IDF vectorizer, then exposes a simple
interface where a user can paste email text and receive a prediction.

This version combines two signals:
1. The ML model's prediction (TF-IDF + Logistic Regression) -- strong
   on lexical/content patterns like urgency language and spam vocabulary.
2. A rule-based URL heuristic (see url_heuristics.py) -- catches a known
   blind spot: calm, professional-sounding phishing emails that mimic
   legitimate business tone but link to a suspicious domain.

Run locally:
    python app.py

Deploy:
    Push this file, requirements.txt, url_heuristics.py, and the
    models/ folder to a Hugging Face Space with the Gradio SDK selected.
"""

import re
import string
import joblib
import gradio as gr

from url_heuristics import analyze_urls, combine_with_model_prediction


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
        return "Please paste an email to analyze.", "", ""

    # --- Signal 1: ML model prediction (content/lexical analysis) ---
    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    model_label = "Phishing" if prediction == 1 else "Legitimate"
    model_confidence = probabilities[prediction] * 100

    # --- Signal 2: URL heuristic (domain/link analysis) ---
    url_analysis = analyze_urls(email_text)

    # --- Combine both signals ---
    final_label, final_confidence, note = combine_with_model_prediction(
        model_label, model_confidence, url_analysis
    )

    confidence_text = f"Confidence: {final_confidence:.1f}%"

    if url_analysis["url_found"]:
        url_detail = f"{url_analysis['verdict']}. {note}"
        if url_analysis["details"]:
            url_detail += " (" + "; ".join(url_analysis["details"]) + ")"
    else:
        url_detail = "No URLs found in this email."

    return final_label, confidence_text, url_detail


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
example_phishing = (
    "Dear Customer, your account has been suspended due to suspicious activity. "
    "Click here immediately to verify your identity and restore access within 24 hours "
    "or your account will be permanently closed. http://paypal-verify-secure.com/login"
)

example_legitimate = (
    "Hi team, just a reminder that our weekly sync is moved to 3 PM tomorrow. "
    "Please review the attached agenda before the meeting. Thanks!"
)

example_tricky = (
    "Hello, as part of our annual compliance review, we ask all employees to confirm "
    "their tax information is up to date. Please visit the link below and enter your "
    "details to ensure uninterrupted payroll processing.\n"
    "https://payroll-confirm.net/update\n"
    "Regards,\nFinance Team"
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
        gr.Textbox(label="URL analysis"),
    ],
    title="Phishing Email Detector",
    description=(
        "Paste any email text below to check whether it is likely phishing or "
        "legitimate. This model combines TF-IDF text classification (97.2% accuracy) "
        "with a rule-based URL analysis to catch professional-sounding phishing "
        "emails that link to suspicious domains."
    ),
    examples=[
        [example_phishing],
        [example_legitimate],
        [example_tricky],
    ],
)

if __name__ == "__main__":
    demo.launch()
