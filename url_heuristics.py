"""
URL Heuristic Module
Adds a rule-based suspicion signal on top of the TF-IDF + ML prediction.

WHY THIS EXISTS:
TF-IDF and word-frequency based models are strong at detecting lexical
patterns common in phishing (urgency language, spam vocabulary, suspicious
phrasing) but have a known blind spot: well-crafted phishing emails that
mimic calm, professional business tone. These emails often contain the
real giveaway not in the wording, but in the URL itself — a suspicious
domain pretending to be a legitimate service.

This module checks any URLs found in the email text against a small set
of common phishing-domain red flags, and combines that signal with the
ML model's prediction rather than replacing it. This mirrors how
production phishing detection systems work in practice: NLP content
analysis plus separate domain/URL reputation signals, not either alone.
"""

import re
from urllib.parse import urlparse


# Common legitimate brand names that phishing domains often try to impersonate.
# Deliberately excludes generic terms like "hr", "it", "security" -- these
# appear naturally in many legitimate internal company URLs and would
# cause false positives. Only specific, well-known external brand names
# are included, since impersonating a recognizable brand is the actual
# phishing pattern this heuristic targets.
WATCHED_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "irs", "payroll",
]

# TLDs disproportionately associated with phishing campaigns
SUSPICIOUS_TLDS = [".win", ".top", ".xyz", ".click", ".loan", ".gq", ".tk", ".ml"]


def extract_urls(text):
    """Find all URLs in the email text."""
    url_pattern = r"https?://[^\s<>\"']+"
    return re.findall(url_pattern, text)


def score_url(url):
    """
    Return a suspicion score (0-3) and a list of reasons for a single URL.
    Higher score = more suspicious.
    """
    reasons = []
    score = 0

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return 0, []

    if not domain:
        return 0, []

    # Flag 1: No HTTPS
    if parsed.scheme != "https":
        score += 1
        reasons.append("does not use HTTPS")

    # Flag 2: IP address used instead of a domain name
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain.split(":")[0]):
        score += 2
        reasons.append("uses a raw IP address instead of a domain name")

    # Flag 3: Brand name combined with hyphens or extra words
    # (e.g. "paypal-verify-secure.com" instead of "paypal.com")
    for brand in WATCHED_BRANDS:
        if brand in domain and domain.replace(brand, "").strip("-.") != "":
            # Brand name present but domain isn't a clean match (e.g. not
            # exactly paypal.com, but paypal-something.com)
            if not re.match(rf"^(www\.)?{brand}\.[a-z.]+$", domain):
                score += 2
                reasons.append(f"impersonates '{brand}' with a non-standard domain")
                break

    # Flag 4: Suspicious top-level domain
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            score += 1
            reasons.append(f"uses an unusual top-level domain ({tld})")
            break

    # Flag 5: Excessive subdomain/hyphen complexity
    # (e.g. "secure-login-account-verify.payroll-confirm.net")
    if domain.count("-") >= 2:
        score += 1
        reasons.append("contains multiple hyphens, a common obfuscation pattern")

    return score, reasons


def analyze_urls(text):
    """
    Analyze all URLs found in the email text.
    Returns a dict with the overall heuristic verdict, score, and explanation.
    """
    urls = extract_urls(text)

    if not urls:
        return {
            "url_found": False,
            "suspicion_score": 0,
            "verdict": "No URLs found in this email.",
            "details": [],
        }

    max_score = 0
    all_reasons = []

    for url in urls:
        score, reasons = score_url(url)
        max_score = max(max_score, score)
        if reasons:
            all_reasons.append(f"{url} — " + "; ".join(reasons))

    if max_score >= 3:
        verdict = "High-risk URL detected"
    elif max_score >= 1:
        verdict = "Moderately suspicious URL detected"
    else:
        verdict = "URL appears standard"

    return {
        "url_found": True,
        "suspicion_score": max_score,
        "verdict": verdict,
        "details": all_reasons,
    }


def combine_with_model_prediction(model_label, model_confidence, url_analysis):
    """
    Combine the ML model's prediction with the URL heuristic signal.

    Logic: the ML model is the primary signal. The URL heuristic acts as
    an override only when it strongly disagrees with the model — this
    avoids the heuristic introducing false positives of its own (e.g.
    flagging a legitimate internal HR portal link just because it has
    a hyphen).
    """
    score = url_analysis["suspicion_score"]

    if model_label == "Phishing":
        # Model already flagged it — URL signal just adds supporting detail
        return "Phishing", model_confidence, "Flagged by content analysis."

    # Model said "Legitimate" — check if the URL heuristic strongly disagrees
    if score >= 3:
        return (
            "Phishing",
            max(model_confidence, 70.0),
            "Content appeared legitimate, but the URL shows strong phishing indicators.",
        )
    elif score >= 1:
        note = "Content appears legitimate, but the URL has some suspicious traits worth a closer look."
        return model_label, model_confidence, note

    return model_label, model_confidence, "No conflicting signals found."
