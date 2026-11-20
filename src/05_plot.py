"""Generate plots from the the sentiment study analysis tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
YEARS = [2009, 2014, 2019, 2024]
PARTIES = ["BJP", "INC", "CPIM", "AIADMK", "DMK", "TMC"]
LABELS = ["negative", "neutral", "positive"]


def save(figure, name):
    for extension in ["png", "svg"]:
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_agreement():
    matrix = pd.read_csv(OUTPUT / "sentiment_confusion.csv", index_col=0).loc[LABELS, LABELS]
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    axis.imshow(matrix, cmap="BuGn", vmin=0)
    axis.set_xticks(range(3), LABELS)
    axis.set_yticks(range(3), LABELS)
    for row in range(3):
        for column in range(3):
            count = matrix.iloc[row, column]
            color = "white" if count > matrix.max().max() * .6 else "#173c40"
            axis.text(column, row, f"{count:,}", ha="center", va="center", color=color)
    axis.set(xlabel="RoBERTa label", ylabel="Pilot lexicon label", title="Where do the two methods agree?")
    save(figure, "01_sentiment_agreement")


def plot_trends():
    data = pd.read_csv(OUTPUT / "by_party_year.csv")
    figure, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10, 9), layout="constrained")
    colors = {"negative": "#a84228", "neutral": "#667782", "positive": "#196a57"}
    for party, axis in zip(PARTIES, axes.flat):
        rows = data[data.party == party].set_index("year").reindex(YEARS)
        axis.set_title({"INC": "Congress", "CPIM": "CPI(M)"}.get(party, party))
        for label in LABELS:
            for method, style in [("lexicon", "--"), ("roberta", "-")]:
                axis.plot(YEARS, rows[f"{method}_{label}_share"] * 100, linestyle=style,
                          color=colors[label], marker="o", markersize=3)
        for year, row in rows.dropna(subset=["distinct_passages"]).iterrows():
            axis.text(year, 2, f"n={int(row.distinct_passages)}", ha="center", fontsize=8)
        axis.set_ylim(0, 100)
        axis.set_xticks(YEARS)
        axis.grid(axis="y", alpha=.15)
    figure.suptitle("Sentiment shares by party and election")
    figure.supylabel("Share of passages (%)")
    figure.supxlabel("Green = positive    Gray = neutral    Orange = negative\nSolid = RoBERTa    Dashed = pilot lexicon")
    save(figure, "02_party_year_trends")


def plot_topics():
    data = pd.read_csv(OUTPUT / "by_topic.csv").nlargest(8, "passage_weight")
    positions = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(9, 5.3), layout="constrained")
    axis.barh(positions - .18, data.lexicon_negative_share * 100, height=.34, label="Pilot lexicon", color="#667782")
    axis.barh(positions + .18, data.roberta_negative_share * 100, height=.34, label="RoBERTa", color="#a84228")
    axis.set_yticks(positions, [f"{topic}  ({weight:,.0f})" for topic, weight in
                                zip(data.topic_label, data.passage_weight)])
    axis.invert_yaxis()
    axis.set(xlabel="Share scored negative (%)", title="Negative sentiment in the eight largest policy categories")
    axis.legend(title="Labels show weighted passage counts")
    save(figure, "03_sentiment_by_topic")

def plot_topic_by_party(topic_code="703"):
    data = pd.read_csv(OUTPUT / "by_party_year_topic.csv")
    selected = data.loc[data["topic_label"].str.startswith(f"{topic_code} -")]
    if selected.empty:
        print(f"Topic {topic_code} has no data in by_party_year_topic.csv.")
        return

    topic = selected["topic_label"].iloc[0]
    colors = {"negative": "#a84228", "neutral": "#667782", "positive": "#196a57"}
    party_names = {"INC": "Congress", "CPIM": "CPI(M)"}
    positions = np.arange(len(YEARS))

    figure, axes = plt.subplots(
        3, 2, sharey=True, figsize=(12, 9), layout="constrained"
    )

    for party, axis in zip(PARTIES, axes.flat):
        rows = selected.loc[selected["party"].eq(party)].set_index("year").reindex(YEARS)
        available = rows["passage_weight"].gt(0).to_numpy()

        for method, offset in [("lexicon", -0.19), ("roberta", 0.19)]:
            bottom = np.zeros(available.sum())
            for label in LABELS:
                shares = rows.loc[available, f"{method}_{label}_share"].to_numpy() * 100
                axis.bar(
                    positions[available] + offset,
                    shares,
                    width=0.33,
                    bottom=bottom,
                    color=colors[label],
                )
                bottom += shares

        ticks = []
        for year, has_data in zip(YEARS, available):
            if has_data:
                n = int(rows.loc[year, "distinct_passages"])
                weight = rows.loc[year, "passage_weight"]
                ticks.append(f"{year}\nn={n}, w={weight:.1f}")
            else:
                ticks.append(f"{year}\n—")

        axis.set_xticks(positions, ticks, fontsize=8)
        axis.set_title(party_names.get(party, party))
        axis.set_xlim(-0.55, len(YEARS) - 0.45)
        axis.set_ylim(0, 100)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.18)

    figure.suptitle(
        f"{topic}\nLexicon on the left, RoBERTa on the right"
        " · n = passages, w = topic weight"
    )
    figure.supylabel("Weighted share of sentiment labels (%)")
    figure.legend(
        handles=[Patch(color=colors[label], label=label.title()) for label in LABELS],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    save(figure, f"04_topic_sentiment_{topic_code}")

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_agreement()
    plot_trends()
    plot_topics()
    plot_topic_by_party("703")


if __name__ == "__main__":
    main()
