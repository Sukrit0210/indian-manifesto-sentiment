"""Compare the two sentiment methods across parties, elections and topics."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"
LABELS = ["negative", "neutral", "positive"]


def summarize(frame, by):
    """Each row contributes its weight to the denominator and to each method."""
    frame = frame.copy()
    frame["weighted_agreement"] = frame["weight"] * (
        frame["lexicon_label"] == frame["contextual_label"]
    )
    counts = ["weighted_agreement"]
    for method, column in [("lexicon", "lexicon_label"), ("roberta", "contextual_label")]:
        for label in LABELS:
            name = f"{method}_{label}"
            frame[name] = frame["weight"] * (frame[column] == label)
            counts.append(name)
    groups = frame.groupby(by, dropna=False)
    table = groups[["weight", *counts]].sum().rename(columns={"weight": "passage_weight"})
    table["distinct_passages"] = groups["passage_id"].nunique()
    table["agreement_share"] = table["weighted_agreement"] / table["passage_weight"]
    for name in counts[1:]:
        table[f"{name}_share"] = table[name] / table["passage_weight"]
    return table.reset_index()

def write_disagreement_audit(scores, passages):
    rows = scores.merge(
        passages[["passage_id", "document_id", "page", "source_url", "text"]],
        on="passage_id",
        validate="one_to_one",
    )
    text = rows["text"].fillna("")

    rows["negation_cue"] = text.str.contains(
        r"\b(?:no|not|never|cannot|without)\b|\b\w+n['’]t\b",
        case=False,
    )
    rows["criticism_cue"] = text.str.contains(
        r"\b(?:failed to|failure of|broken promises?|neglected|mismanaged)\b",
        case=False,
    )
    commitment = text.str.contains(
        r"\b(?:we (?:will|shall|pledge to|promise to|commit to)|"
        r"our government (?:will|shall))\b",
        case=False,
    )
    problem = text.str.contains(
        r"\b(?:poverty|unemployment|corruption|violence|discrimination|"
        r"inflation|hunger|pollution|inequality|shortages?)\b",
        case=False,
    )
    rows["commitment_and_problem_cue"] = commitment & problem

    cues = ["negation_cue", "criticism_cue", "commitment_and_problem_cue"]
    rows["no_tracked_cue"] = ~rows[cues].any(axis=1)
    rows["disagree"] = rows["lexicon_label"] != rows["contextual_label"]
    rows["roberta_confidence"] = rows[["p_negative", "p_neutral", "p_positive"]].max(axis=1)

    fields = [
        "passage_id", "document_id", "party", "year", "page", "source_url",
        "lexicon_label", "contextual_label", "roberta_confidence",
        *cues, "no_tracked_cue", "text",
    ]
    rows.loc[rows["disagree"], fields].to_csv(
        OUTPUT / "sentiment_disagreements.csv", index=False
    )

    summary = []
    examples = []
    groups = [("all_passages", rows)]
    groups.extend((cue, rows.loc[rows[cue]]) for cue in [*cues, "no_tracked_cue"])

    label_columns = ["lexicon_label", "contextual_label"]
    for cue, group in groups:
        different = group.loc[group["disagree"]]
        summary.append({
            "cue": cue,
            "passages": len(group),
            "disagreements": len(different),
            "disagreement_share": len(different) / len(group) if len(group) else None,
        })
        if cue != "all_passages":
            sample = different.sort_values("passage_id").drop_duplicates(
                label_columns + ["party", "year"]
            )
            sample = sample.groupby(label_columns).head(3)
            examples.append(sample[fields].assign(cue=cue))

    pd.DataFrame(summary).to_csv(OUTPUT / "disagreement_by_cue.csv", index=False)
    pd.concat(examples, ignore_index=True)[["cue", *fields]].to_csv(
        OUTPUT / "disagreement_cue_examples.csv", index=False
    )

def main():
    scores = pd.read_csv(OUTPUT / "scores.csv")
    passages = pd.read_csv(DATA / "master_passages.csv")
    topics = pd.read_csv(OUTPUT / "passage_topics.csv")
    if scores["passage_id"].duplicated().any() or set(scores["passage_id"]) != set(passages["passage_id"]):
        raise ValueError("Score each master passage once before running analysis")
    for method in ["lexicon_label", "contextual_label"]:
        if not scores[method].isin(LABELS).all():
            raise ValueError(f"Unexpected sentiment labels in {method}")
    weights = topics.groupby("passage_id")["weight"].sum()
    if not ((weights - 1.0).abs() < 1e-9).all():
        raise ValueError("Topic weights must sum to one within every covered passage")
    if not topics["passage_id"].isin(scores["passage_id"]).all():
        raise ValueError("Rebuild the topic assignments for the current passages")

    scores["weight"] = 1.0
    summarize(scores, ["party", "year"]).to_csv(OUTPUT / "by_party_year.csv", index=False)
    topic_scores = scores.drop(columns="weight").merge(topics, on="passage_id", validate="one_to_many")
    summarize(topic_scores, ["party", "year", "topic_label"]).to_csv(
        OUTPUT / "by_party_year_topic.csv", index=False
    )
    summarize(topic_scores, ["topic_label"]).to_csv(OUTPUT / "by_topic.csv", index=False)

    coverage = scores.assign(has_topic=scores["passage_id"].isin(weights.index))
    coverage.groupby(["party", "year"])["has_topic"].agg(["count", "sum"]).rename(
        columns={"count": "passages", "sum": "passages_with_topics"}
    ).to_csv(OUTPUT / "topic_coverage.csv")

    matrix = pd.crosstab(scores["lexicon_label"], scores["contextual_label"])
    matrix.reindex(index=LABELS, columns=LABELS, fill_value=0).to_csv(
        OUTPUT / "sentiment_confusion.csv"
    )

    # Sample up to four examples for each label combination for a qualitative check.
    cases = []
    for _, group in scores.groupby(["lexicon_label", "contextual_label"]):
        cases.append(group.sample(n=min(len(group), 4), random_state=2024))
    examples = pd.concat(cases).merge(
        passages[["passage_id", "document_id", "page", "source_url", "text"]],
        on="passage_id",
        validate="one_to_one",
    )
    fields = [
        "passage_id", "document_id", "party", "year", "page", "source_url",
        "lexicon_label", "contextual_label", "text",
    ]
    examples[fields].sort_values(
        ["lexicon_label", "contextual_label", "passage_id"]
    ).to_csv(OUTPUT / "comparison_examples.csv", index=False)

    write_disagreement_audit(scores, passages)


if __name__ == "__main__":
    main()
