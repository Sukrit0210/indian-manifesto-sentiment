"""Assign policy topics to passages using short pieces of text within them."""
from collections import Counter, defaultdict

import pysbd
from boundaries import ACTION_START, WORDS, spans
from common import DATA, OUTPUTS, read, write
from topic_model import classify_topics

SENTENCES = pysbd.Segmenter(language="en", clean=False, char_span=True)


def segments(passage):
    """Apply the verified topic input boundaries to a single passage."""
    text = passage["text"]
    for sentence in SENTENCES.segment(text):
        for first, last in spans(sentence.sent):
            unit = text[sentence.start + first : sentence.start + last].strip()
            if not unit:
                continue
            words = len(WORDS.findall(unit))
            action = ACTION_START.match(unit)
            possible_heading = unit.istitle() and words < 12 and not action
            if (words < 3 or possible_heading) and not action:
                continue
            yield {"passage_id": passage["passage_id"], "text": unit}


def main():
    passages = read(DATA / "master_passages.csv")
    topic_inputs = [unit for passage in passages for unit in segments(passage)]
    assignments = classify_topics(topic_inputs)
    by_passage = defaultdict(Counter)
    for row, topic in zip(topic_inputs, assignments, strict=True):
        by_passage[row["passage_id"]][topic] += 1
    write(
        OUTPUTS / "passage_topics.csv",
        [
            dict(passage_id=passage_id, topic_label=topic,
                 segments=count, weight=count / counts.total())
            for passage_id, counts in by_passage.items()
            for topic, count in sorted(counts.items())
        ],
    )
    print(f"Aggregated {len(assignments):,} topic assignments across {len(by_passage):,} passages")


if __name__ == "__main__":
    main()
