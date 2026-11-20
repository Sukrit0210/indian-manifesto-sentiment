import re

POSITIVE_WORDS = {
    "improve",
    "improved",
    "strengthen",
    "expand",
    "support",
    "protect",
    "affordable",
    "quality",
    "secure",
    "safe",
    "growth",
    "empower",
    "opportunity",
    "transparent",
    "sustainable",
    "free",
    "benefit",
    "welfare",
    "justice",
    "development",
    "progress",
    "inclusive",
}

NEGATIVE_WORDS = {
    "crisis",
    "poverty",
    "unemployment",
    "corruption",
    "violence",
    "discrimination",
    "inflation",
    "attack",
    "deprivation",
    "failure",
    "fraud",
    "unsafe",
    "hunger",
    "malnutrition",
    "pollution",
    "disaster",
    "exploitation",
    "inequality",
    "injustice",
    "weak",
    "decline",
}


def tokenize(text):
    return re.findall(r"\b[a-z][a-z]+\b", str(text).lower())


def sentiment_score(text):
    words = tokenize(text)
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    # Adding 1 prevents division by zero for neutral paragraphs.
    score = (positive_count - negative_count) / (positive_count + negative_count + 1)

    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return positive_count, negative_count, score, label
