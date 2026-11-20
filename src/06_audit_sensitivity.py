"""Measure how controlled text changes affect the two sentiment methods."""
from collections import defaultdict
import argparse
import hashlib
from itertools import combinations
import random
import re

import matplotlib.pyplot as plt

from common import CACHE, DATA, OUTPUTS, read, write
from lexicon import sentiment_score
from torch_inference import infer_sentiment

DEST = OUTPUTS / "sensitivity"
LABELS = ("negative", "neutral", "positive")
PARTIES = re.compile(r"\b(BJP|Congress)\b", re.I)
NOISE = {"o": "0", "i": "1", "l": "1", "m": "rn", " ": ""}
PER_DOCUMENT = 20


def seed(identifier):
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")


def corrupt(text, identifier, rate):
    rng = random.Random(seed(identifier))
    result = []
    for char in text:
        if char.lower() in NOISE and rng.random() < rate:
            char = NOISE[char.lower()]
        result.append(char)
    return "".join(result)


def make_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["document_id"]].append(row)
    cases = []
    for document, group in sorted(groups.items()):
        sample = sorted(group, key=lambda r: seed(r["passage_id"]))
        native = [r for r in sample if r["extraction"] == "pdf_text"]
        named = [r for r in sample if PARTIES.search(r["text"])]
        for row in native[:PER_DOCUMENT]:
            for rate in (0.01, 0.03, 0.05):
                altered = corrupt(row["text"], row["passage_id"], rate)
                cases.append((document, row["passage_id"], "simulated_ocr", rate, row["text"], altered))
        for row in named[:PER_DOCUMENT]:
            altered = PARTIES.sub(lambda m: "Congress" if m[0].lower() == "bjp" else "BJP", row["text"])
            cases.append((document, row["passage_id"], "party_name_swap", 0, row["text"], altered))
    for i, phrase in enumerate(("the scheme is effective", "the reform is successful",
                                "the service is reliable", "the outcome is satisfactory"), 1):
        text = phrase.capitalize() + "."
        cases.append(("constructed", f"negation-{i}", "negation", 0, text, text.replace(" is ", " is not ")))
    return cases


def compare_cases(cases):
    texts = list(dict.fromkeys(text for case in cases for text in case[-2:]))
    scores = {}
    for text, probs in zip(texts, infer_sentiment(texts), strict=True):
        _, _, score, label = sentiment_score(text)
        scores[text] = {"lexicon": (label, score),
                        "roberta": (LABELS[int(probs.argmax())], float(probs[2] - probs[0]))}
    details = []
    for doc, passage, test, rate, original, altered in cases:
        for method, (label, score) in scores[original].items():
            after, value = scores[altered][method]
            details.append(dict(document_id=doc, passage_id=passage, test=test,
                rate_per_eligible_character=rate, method=method, original=original, altered=altered,
                text_changed=int(original != altered), original_label=label, altered_label=after,
                label_changed=int(label != after), score_delta=value-score))
    return details


def summarize(details):
    groups = defaultdict(list)
    for row in details:
        groups[row["test"], row["rate_per_eligible_character"], row["method"]].append(row)
    summaries = []
    for (test, rate, method), group in sorted(groups.items()):
        n = len(group)
        summaries.append(dict(test=test, rate_per_eligible_character=rate, method=method, cases=n,
            texts_changed=sum(r["text_changed"] for r in group),
            flip_share=sum(r["label_changed"] for r in group)/n,
            mean_signed_score_change=sum(r["score_delta"] for r in group)/n,
            lower_score_count=sum(r["score_delta"] < 0 for r in group)))
    return summaries


def compare_orderings():
    groups = defaultdict(list)
    for row in read(OUTPUTS / "by_party_year_topic.csv"):
        groups[row["year"], row["topic_label"]].append(row)
    result = []
    for (year, topic), group in sorted(groups.items()):
        for a, b in combinations(sorted(group, key=lambda r: r["party"]), 2):
            gaps = [float(a[m + "_negative_share"]) - float(b[m + "_negative_share"])
                    for m in ("lexicon", "roberta")]
            result.append(dict(year=year, topic=topic, party_a=a["party"], party_b=b["party"],
                weight_a=a["passage_weight"], weight_b=b["passage_weight"],
                lexicon_gap=gaps[0], roberta_gap=gaps[1],
                ordering="tie_in_either" if min(map(abs, gaps)) < 1e-10 else
                         "reversed" if gaps[0]*gaps[1] < 0 else "same"))
    fields = ["year","topic","party_a","party_b","weight_a","weight_b","lexicon_gap","roberta_gap","ordering"]
    write(DEST / "party_orderings.csv", result, fields=fields)
    counts = []
    for minimum in (0, 5, 10):
        eligible = [r for r in result if min(float(r["weight_a"]), float(r["weight_b"])) >= minimum]
        counts.append(dict(minimum_weight_per_party=minimum, comparisons=len(eligible),
                           reversed=sum(r["ordering"] == "reversed" for r in eligible),
                           same=sum(r["ordering"] == "same" for r in eligible),
                           ties=sum(r["ordering"] == "tie_in_either" for r in eligible)))
    write(DEST / "ordering_summary.csv", counts)


def third_model(rows):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = "facebook/bart-large-mnli"
    revision = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    options = dict(revision=revision, cache_dir=CACHE / "hf")
    tokenizer = AutoTokenizer.from_pretrained(repo, **options)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(repo, **options).to(device).eval()
    entail = next(i for i, name in model.config.id2label.items() if name.lower() == "entailment")
    scores = {r["passage_id"]: r for r in read(OUTPUTS / "scores.csv")}
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    cases = []
    hypotheses = [f"The tone of this passage is {label}." for label in LABELS]
    with torch.inference_mode():
        for doc, group in sorted(by_document.items()):
            for row in sorted(group, key=lambda r: seed(r["passage_id"]))[:PER_DOCUMENT]:
                text = row["text"]
                batch = tokenizer([text]*3, hypotheses, truncation="only_first", max_length=512,
                                  padding=True, return_tensors="pt").to(device)
                ent = model(**batch).logits[:, entail]
                saved = scores[row["passage_id"]]
                cases.append(dict(document_id=doc, passage_id=row["passage_id"], party=row["party"],
                    year=row["year"], lexicon=saved["lexicon_label"], roberta=saved["contextual_label"],
                    bart=LABELS[int(ent.argmax())], max_input_tokens=512, checkpoint=revision))
    write(DEST / "three_method_cases.csv", cases)
    summary = []
    for a, b in combinations(("lexicon", "roberta", "bart"), 2):
        agreed = sum(r[a] == r[b] for r in cases)
        summary.append(dict(method_a=a, method_b=b, passages=len(cases),
                            agreed=agreed, share=agreed / len(cases)))
    write(DEST / "three_method_agreement.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-model", action="store_true")
    args = parser.parse_args()
    rows = read(DATA / "master_passages.csv")
    details = compare_cases(make_cases(rows))
    summary = summarize(details)
    write(DEST / "perturbation_cases.csv", details)
    write(DEST / "perturbation_summary.csv", summary)
    compare_orderings()
    fig, ax = plt.subplots()
    for method in ("lexicon", "roberta"):
        points = [r for r in summary if r["test"] == "simulated_ocr" and r["method"] == method]
        ax.plot([100*r["rate_per_eligible_character"] for r in points],
                [100*r["flip_share"] for r in points], marker="o", label=method)
    ax.set(xlabel="Corruption probability per eligible character (%)",
           ylabel="Passages with a changed sentiment label (%)", title="Sensitivity to simulated OCR errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DEST / "ocr_sensitivity.png", dpi=160)
    plt.close(fig)
    if args.third_model:
        third_model(rows)
    print("Saved sentiment sensitivity results to", DEST)


if __name__ == "__main__":
    main()
