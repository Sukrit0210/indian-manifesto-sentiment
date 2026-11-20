"""Compare the original word-count baseline and contextual polarity on the same passages."""


from common import DATA, OUTPUTS, read, write
from lexicon import sentiment_score
from torch_inference import infer_sentiment

LABELS = ["negative", "neutral", "positive"]


def main():
    rows = read(DATA / "master_passages.csv")
    probabilities = infer_sentiment([row["text"] for row in rows])
    results = []
    for row, probabilities_for_passage in zip(rows, probabilities, strict=True):
        positive_count, negative_count, score, base = sentiment_score(row["text"])
        label = LABELS[probabilities_for_passage.argmax()]
        meta = {k: row[k] for k in ("passage_id", "party", "year", "policy_area")}
        results.append(
            dict(
                **meta,
                positive_words=positive_count,
                negative_words=negative_count,
                lexicon_score=score,
                lexicon_label=base,
                contextual_label=label,
                **{
                    "p_" + k: float(v)
                    for k, v in zip(LABELS, probabilities_for_passage)
                },
            )
        )
    write(OUTPUTS / "scores.csv", results)
    print(f"Saved {len(results):,} sentiment scores to {OUTPUTS}")


if __name__ == "__main__":
    main()
