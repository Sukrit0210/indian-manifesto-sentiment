"""Run a larger common-sample BART comparison and source-based negation tests."""
import argparse
from collections import Counter, defaultdict
import importlib
from itertools import combinations
import re
from statistics import median

import pandas as pd

from common import CACHE, DATA, OUTPUTS, read, write
from lexicon import sentiment_score
from torch_inference import infer_sentiment

DEST = OUTPUTS / "sensitivity" / "expanded"
LABELS = ("negative", "neutral", "positive")
POSITIVE = "effective|successful|beneficial|helpful|reliable|satisfactory|welcome|commendable|adequate|sufficient|fair|encouraging|impressive|excellent|good|important"
NEGATIVE = "ineffective|unsuccessful|harmful|unreliable|unsatisfactory|inadequate|insufficient|unfair|unacceptable|disappointing|poor|bad|wrong|corrupt|damaging"
EVALUATION = re.compile(
    r"\b(?P<copula>is|are|was|were)\s+(?:(?:very|highly|clearly|quite|particularly)\s+)?"
    r"(?:(?P<positive>" + POSITIVE + r")|(?P<negative>" + NEGATIVE + r"))(?=\s*[,.;!?]|$)",
    re.I,
)
AMBIGUOUS = re.compile(r"\b(?:not|never|hardly|neither|if|whether|although|however)\b", re.I)
CONTEXT_PROMPTS = (
    ("This proposal is effective.", "less_positive"),
    ("This proposal is ineffective.", "more_positive"),
    ("We believe this approach is worthwhile.", "less_positive"),
    ("We believe this approach is harmful.", "more_positive"),
    ("Overall, this plan is satisfactory.", "less_positive"),
)


def negation_candidate(row):
    text = row["text"]
    hits = list(EVALUATION.finditer(text))
    if len(hits) != 1 or text.count('"') or len(text.split()) > 100:
        return None
    match = hits[0]
    if AMBIGUOUS.search(text[max(0, match.start() - 48):match.start()]):
        return None
    end = match.end("copula")
    return dict(document_id=row["document_id"], passage_id=row["passage_id"], party=row["party"],
                year=row["year"], page=row["page"], case_type="direct_source_edit",
                source_text=text, original=text,
                altered=text[:end] + " not" + text[end:],
                edit_offset=end, edit_text=" not", cue=match.group(),
                expected_direction="less_positive" if match["positive"] else "more_positive")


def context_case(row, index):
    """Supply an evaluative clause when safe direct edits are scarce."""
    clause, direction = CONTEXT_PROMPTS[index]
    source = row["text"].strip()
    prefix = source + (" " if source.endswith((".", "?", "!")) else ". ")
    end = len(prefix) + clause.index(" is ") + 3
    original = prefix + clause
    return dict(document_id=row["document_id"], passage_id=row["passage_id"], party=row["party"],
                year=row["year"], page=row["page"], case_type="constructed_clause_with_source_context",
                source_text=row["text"], original=original,
                altered=original[:end] + " not" + original[end:], edit_offset=end,
                edit_text=" not", cue=clause, expected_direction=direction)


def negation_audit(rows, seed):
    by_document = defaultdict(list)
    candidates = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
        candidate = negation_candidate(row)
        if candidate:
            candidates[row["document_id"]].append(candidate)
    sample, coverage = [], []
    for doc, group in sorted(by_document.items()):
        pool = sorted(candidates[doc], key=lambda r: seed(r["passage_id"]))
        queues = {direction: [r for r in pool if r["expected_direction"] == direction]
                  for direction in ("less_positive", "more_positive")}
        picks = [r for pair in zip(*queues.values()) for r in pair][:5]
        used = {r["passage_id"] for r in picks}
        picks.extend([r for r in pool if r["passage_id"] not in used][:5 - len(picks)])
        used = {r["passage_id"] for r in picks}
        direct = len(picks)
        context = sorted(
            (r for r in group if r["passage_id"] not in used and 15 <= int(r["word_count"]) <= 60
             and int(r["text"].count("\n")) <= 1),
            key=lambda r: seed(r["passage_id"]),
        )
        picks.extend(context_case(r, i) for i, r in enumerate(context[:5-direct]))
        sample.extend(picks)
        coverage.append(dict(document_id=doc, party=group[0]["party"], year=group[0]["year"],
                             eligible_direct=len(pool), direct_selected=direct,
                             constructed_context_selected=len(picks)-direct, selected=len(picks), target=5,
                             less_positive=sum(r["expected_direction"] == "less_positive" for r in picks),
                             more_positive=sum(r["expected_direction"] == "more_positive" for r in picks)))
    write(DEST / "negation_coverage.csv", coverage)
    texts = [text for row in sample for text in (row["original"], row["altered"])]
    if not texts:
        raise ValueError("No unambiguous negation candidates were found in this corpus.")
    contextual = iter(infer_sentiment(texts))
    results = []
    for row in sample:
        before, after = next(contextual), next(contextual)
        _, _, lex_before, lex_label = sentiment_score(row["original"])
        _, _, lex_after, lex_edited = sentiment_score(row["altered"])
        for method, baseline, edited, label_a, label_b in (
            ("lexicon", lex_before, lex_after, lex_label, lex_edited),
            ("roberta", float(before[2] - before[0]), float(after[2] - after[0]),
             LABELS[int(before.argmax())], LABELS[int(after.argmax())]),
        ):
            delta = edited - baseline
            movement = "less_positive" if delta < -1e-8 else "more_positive" if delta > 1e-8 else "unchanged"
            results.append(dict(row, method=method, original_label=label_a, altered_label=label_b,
                                score_delta=delta, movement=movement,
                                matches_direction=int(movement == row["expected_direction"])))
    write(DEST / "negation_cases.csv", results)
    counts = []
    for method in ("lexicon", "roberta"):
        for case_type in ("direct_source_edit", "constructed_clause_with_source_context"):
            for direction in ("all", "less_positive", "more_positive"):
                subset = [r for r in results if r["method"] == method and r["case_type"] == case_type
                          and (direction == "all" or r["expected_direction"] == direction)]
                counts.append(dict(method=method, case_type=case_type, expected_direction=direction,
                                   cases=len(subset), correct_direction=sum(r["matches_direction"] for r in subset),
                                   unchanged=sum(r["movement"] == "unchanged" for r in subset),
                                   flips=sum(r["original_label"] != r["altered_label"] for r in subset)))
    write(DEST / "negation_summary.csv", counts)


def group_sensitivity(rows):
    source = {r["document_id"]: r for r in rows}
    groups = defaultdict(list)
    for row in read(OUTPUTS / "sensitivity/perturbation_cases.csv"):
        if row["test"] == "negation":
            continue
        doc = source[row["document_id"]]
        for grouping, value in (("document", row["document_id"]), ("party", doc["party"]), ("year", doc["year"])):
            groups[grouping, value, row["test"], row["rate_per_eligible_character"], row["method"]].append(row)
    write(DEST / "perturbations_by_group.csv", [
        dict(grouping=g, group=value, test=test, rate=rate, method=method, cases=len(items),
             label_changes=sum(int(r["label_changed"]) for r in items),
             flip_share=sum(int(r["label_changed"]) for r in items) / len(items))
        for (g, value, test, rate, method), items in sorted(groups.items())
    ])
    orders = read(OUTPUTS / "sensitivity/party_orderings.csv")
    summaries = []
    for minimum in (0, 5, 10):
        for group in ("all", "reversed", "same", "tie_in_either"):
            subset = [r for r in orders if min(float(r["weight_a"]), float(r["weight_b"])) >= minimum
                      and (group == "all" or r["ordering"] == group)]
            gaps = [abs(float(r["lexicon_gap"]) - float(r["roberta_gap"])) * 100 for r in subset]
            summaries.append(dict(minimum_weight=minimum, ordering=group, comparisons=len(gaps),
                                  median_abs_gap_difference_points=median(gaps) if gaps else "",
                                  mean_abs_gap_difference_points=sum(gaps)/len(gaps) if gaps else ""))
    write(DEST / "ordering_gap_sizes.csv", summaries)


def sample_passages(rows, seed):
    topics = {}
    for r in read(OUTPUTS / "passage_topics.csv"):
        proposal = (float(r["weight"]), r["topic_label"])
        if proposal > topics.get(r["passage_id"], (0.0, "")):
            topics[r["passage_id"]] = proposal
    cues = importlib.import_module("04_analyze").disagreement_cues(pd.Series([r["text"] for r in rows]))
    groups = defaultdict(lambda: defaultdict(list))
    for row, flags in zip(rows, cues.itertuples(index=False, name=None), strict=True):
        label = topics.get(row["passage_id"], (0, "unassigned"))[1]
        cue = "+".join(name for name, present in zip(cues.columns[:3], flags[:3]) if present) or "none"
        groups[row["document_id"]][label, cue].append(row)
    sample = []
    for _, strata in sorted(groups.items()):
        population = sum(map(len, strata.values()))
        size = min(50, population)
        quotas = {key: size * len(items) / population for key, items in strata.items()}
        extra = size - sum(int(q) for q in quotas.values())
        bonuses = set(sorted(quotas, key=lambda key: (-(quotas[key] % 1), key))[:extra])
        for key, items in sorted(strata.items()):
            selected = sorted(items, key=lambda r: seed(r["passage_id"]))[:int(quotas[key]) + (key in bonuses)]
            sample.extend(dict(r, dominant_topic=key[0], cue_group=key[1]) for r in selected)
    return sample


def bart_comparison(rows, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo, revision = "facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    options = dict(revision=revision, cache_dir=CACHE / "hf")
    tokenizer = AutoTokenizer.from_pretrained(repo, **options)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(repo, **options).to(device).eval()
    entailment = next(i for i, label in model.config.id2label.items() if label.lower() == "entailment")
    sample = sample_passages(rows, seed)
    hypotheses = [f"The tone of this passage is {label}." for label in LABELS]
    hypotheses.append("This passage uses factual or procedural language without clear praise or criticism.")
    saved = {r["passage_id"]: r for r in read(OUTPUTS / "scores.csv")}
    results = []
    with torch.inference_mode():
        for offset in range(0, len(sample), 8):
            group = sample[offset:offset + 8]
            batch = tokenizer([r["text"] for r in group for _ in hypotheses], hypotheses * len(group),
                              truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(device)
            scores = model(**batch).logits[:, entailment].reshape(len(group), len(hypotheses)).cpu()
            for row, logits in zip(group, scores, strict=True):
                baseline = saved[row["passage_id"]]
                results.append(dict(document_id=row["document_id"], passage_id=row["passage_id"],
                    party=row["party"], year=row["year"], dominant_topic=row["dominant_topic"],
                    cue_group=row["cue_group"], lexicon=baseline["lexicon_label"], roberta=baseline["contextual_label"],
                    bart=LABELS[int(logits[:3].argmax())], bart_neutral_explicit=LABELS[int(logits[[0, 3, 2]].argmax())]))
    write(DEST / "bart_cases.csv", results)
    methods = ("lexicon", "roberta", "bart", "bart_neutral_explicit")
    write(DEST / "bart_label_counts.csv", [
        dict(method=method, label=label, passages=sum(r[method] == label for r in results))
        for method in methods for label in LABELS
    ])
    cohorts = defaultdict(list)
    cohorts["all"].extend(results)
    for row in results:
        cohorts["cue:" + row["cue_group"]].append(row)
    write(DEST / "bart_agreement.csv", [
        dict(group=group, method_a=a, method_b=b, passages=len(items),
             agreed=sum(r[a] == r[b] for r in items), share=sum(r[a] == r[b] for r in items)/len(items))
        for group, items in sorted(cohorts.items()) for a, b in combinations(methods, 2)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-third-model", action="store_true")
    args = parser.parse_args()
    seed = importlib.import_module("06_audit_sensitivity").seed
    rows = read(DATA / "master_passages.csv")
    group_sensitivity(rows)
    negation_audit(rows, seed)
    if not args.no_third_model:
        bart_comparison(rows, seed)
    print("Expanded sentiment audit saved to", DEST)


if __name__ == "__main__":
    main()
