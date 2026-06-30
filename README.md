---
title: Phishing Email Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.9.0"
app_file: app.py
pinned: false
---
# Phishing Email Detection using NLP and Machine Learning

A machine learning system that classifies emails as phishing or legitimate using Natural Language Processing and supervised learning, achieving **97.2% classification accuracy** with **56% fewer false positives** using Logistic Regression compared to a Random Forest baseline.

**[Live Demo](#)** &nbsp;·&nbsp; **[Source Code](#)** &nbsp;·&nbsp; Built by [Divvela Hemarshini](https://www.linkedin.com/in/divvelahemarshini)

---

## Overview

Phishing emails remain one of the most common attack vectors in cybersecurity, and manual detection does not scale. This project builds an end-to-end NLP pipeline that processes raw email text and classifies it as phishing or legitimate, comparing two supervised learning approaches to identify the strongest-performing model for real-world deployment.

The system was built independently from data preprocessing through model evaluation and deployment, with a focus on minimizing false positives — a critical requirement for any spam filter, since flagging legitimate emails as phishing directly disrupts user trust.

## Key Results

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | 96.6% | 97.2% |
| Precision | 94.8% | 97.6% |
| Recall | 96.2% | 94.8% |
| F1-score | 95.5% | 96.2% |
| False positives (test set) | 103 | 45 |

**Headline result:** Logistic Regression achieved 97.2% accuracy with 56% fewer false positives than the Random Forest baseline, making it the deployed model.

## How It Works

The pipeline takes raw email text through five stages:

1. **Data loading and cleaning** — raw emails are loaded, deduplicated, and normalized for consistent processing
2. **Text preprocessing** — lowercasing, URL removal, punctuation and digit stripping to clean the raw email body
3. **Feature extraction** — TF-IDF vectorization with English stop-word removal and unigram/bigram features converts cleaned text into numerical vectors
4. **Model training** — Logistic Regression and Random Forest classifiers are trained on the TF-IDF feature matrix
5. **Evaluation** — both models are benchmarked using accuracy, precision, recall, F1-score, and confusion matrix analysis to select the strongest performer

## Tech Stack

`Python` `Scikit-learn` `Pandas` `NumPy` `NLP` `TF-IDF` `Gradio` `Matplotlib` `Seaborn`

## Dataset

The model is trained on a labeled phishing email dataset (`Phishing_Email.csv`, 18,650 emails) sourced from a public Kaggle dataset, containing raw email text labeled as either "Phishing Email" or "Safe Email." After deduplication and removing missing values, 17,537 emails remained for training (10,979 legitimate, 6,558 phishing).

## Project Structure

```
Phishing_Email_Detection_using_NLP_and_Machine_Learning/
├── data/
│   └── phishing_emails.csv
├── models/
│   ├── phishing_model.joblib
│   └── tfidf_vectorizer.joblib
├── notebooks/
│   └── phishing_email_detection.ipynb
├── results/
│   ├── random_forest_confusion_matrix.png
│   └── logistic_regression_confusion_matrix.png
├── app.py
├── model_training.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/divvelahemarshini/Phishing_Email_Detection_using_NLP_and_Machine_Learning.git
cd Phishing_Email_Detection_using_NLP_and_Machine_Learning
pip install -r requirements.txt
```

### Train the model

Place your dataset inside `data/phishing_emails.csv`, then run:

```bash
python model_training.py
```

This trains both models, prints evaluation metrics, saves confusion matrix plots to `results/`, and saves the best-performing model and vectorizer to `models/`.

### Run the demo locally

```bash
python app.py
```

This launches a local Gradio interface where you can paste email text and get a live prediction.

## Live Demo

Try the deployed model on Hugging Face Spaces: **[Live Demo](#)**

Paste any email text and get an instant phishing/legitimate prediction with a confidence score.

## Future Enhancements

- Extend the pipeline with deep learning approaches (LSTM, Transformer-based models) to capture richer semantic patterns
- Add model interpretability using SHAP or LIME to explain individual predictions
- Expand the dataset with additional phishing campaign types for broader generalization

## License

This project is released under the [MIT License](LICENSE).

## Author

**Divvela Hemarshini**
AI & Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/divvelahemarshini) &nbsp;·&nbsp; [GitHub](https://github.com/divvelahemarshini)
