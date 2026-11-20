
# Sentiment in Indian Election Manifestos

This project compares the original word-count lexicon and pretrained RoBERTa on 9,572 passages from 21 English Lok Sabha manifestos. ManifestoBERTa adds policy categories so the sentiment comparison can be made within the same issue across parties and elections. The corpus covers 2009, 2014, 2019 and 2024. [Original pilot](https://github.com/Sukrit0210/indian-manifesto-text-mining) · [Report](REPORT.md) · [Source inventory](data/sources.csv)

## Changes from the original pilot

| Area | What changed |
|---|---|
| Corpus | Expanded the balanced 832-passage sample of four parties in 2024 to 9,572 retained passages from six parties across four elections. Passage counts now reflect the included editions. |
| Source tracking | Added PDF checksums, page numbers, source URLs, English OCR where needed, and a stable `passage_id` for each extracted passage. |
| Sentiment | Kept the original word lists, score, and thresholds as the baseline. Added pretrained PyTorch RoBERTa and stored both methods’ labels for every passage. |
| Policy categories | The pilot explored topics with LDA and NMF. This study uses ManifestoBERTa’s 56 policy categories to compare the same issue across parties and elections. Short-unit topic assignments roll up to passage weights. |
| `src/04_analyze.py` | Added the disagreement audit. It writes `disagreement_by_cue.csv`, `sentiment_disagreements.csv`, and `disagreement_cue_examples.csv` for negation, criticism, and passages with commitment and problem language. |
| `src/05_plot.py` | Added the side-by-side agriculture chart with all three sentiment labels for each party and election. It saves `04_topic_sentiment_703` in PNG and SVG. |
| Local setup | Updated the Windows launcher to prepare a project-local Python 3.14 environment and install the CUDA-enabled PyTorch build before running the stages. The current GPU run reproduced the reference corpus and sentiment labels. |

## Run on Windows

Extract the source PDFs directly into `data/raw/`. Use 64-bit Python 3.14. For the verified GPU run, the launcher creates `.venv314/`, installs PyTorch 2.10.0 with CUDA 12.6, installs the pinned dependencies, and tests CUDA. Open the project folder in VS Code and run `run_all.py`, or run the following from PowerShell in the project folder.

```powershell
py -3.14 run_all.py
```

The first run needs internet access for the dependencies and pinned model files. It stores the weights in `.cache/models/` and verifies the PDF and checkpoint files against their recorded checksums. If your PDF files are somewhere else, run `py -3.14 run_all.py --pdf-dir "PATH_TO_PDFS"`. [Edition notes](data/edition_notes.md) · [Model revisions and hashes](data/models.json)

| Stage | Script | Result |
|---|---|---|
| 01 | [Build the dataset](src/01_build_dataset.py) | Master passages and source counts in `data/processed/`. |
| 02 | [Score sentiment](src/02_score_sentiment.py) | Original lexicon scores and RoBERTa labels and probabilities for each passage. |
| 03 | [Assign policy topics](src/03_assign_topics.py) | ManifestoBERTa topic weights within each passage. |
| 04 | [Analyze](src/04_analyze.py) | Sentiment summaries, agreement counts, topic coverage, and disagreement examples. |
| 05 | [Plot](src/05_plot.py) | Four figures in PNG and SVG, including the agriculture breakdown. |

To regenerate the analysis and plots after stages 01–03, run:

```powershell
py -3.14 run_all.py --start 4
```

To run an individual script in VS Code, select `.venv314\Scripts\python.exe` as the interpreter after setup.

## Find the results

| Files | What they contain |
|---|---|
| [Master passages](data/processed/master_passages.csv) · [Source counts](data/processed/source_counts.csv) | The rebuilt corpus and counts by source PDF. The master table also feeds the separate pledge study. |
| [Scores](outputs/scores.csv) · [Passage topics](outputs/passage_topics.csv) | Both sentiment decisions for each passage, plus ManifestoBERTa’s policy categories and fractional topic weights. Join them by `passage_id`. |
| [Party and year](outputs/by_party_year.csv) · [Party, year and topic](outputs/by_party_year_topic.csv) · [Topic totals](outputs/by_topic.csv) · [Topic coverage](outputs/topic_coverage.csv) | Counts, agreement, and sentiment shares. A passage can contribute fractional weight to multiple topics. |
| [Agreement counts](outputs/sentiment_confusion.csv) · [Comparison examples](outputs/comparison_examples.csv) | A three-by-three table of the two methods’ labels and examples from the label combinations. |
| [Disagreement rates by cue](outputs/disagreement_by_cue.csv) · [All disagreements](outputs/sentiment_disagreements.csv) · [Cue examples](outputs/disagreement_cue_examples.csv) | A source-linked audit for markers of negation, criticism, and passages containing both commitment and problem language. |

The four figures show [label agreement](outputs/figures/01_sentiment_agreement.png), [party-year trends](outputs/figures/02_party_year_trends.png), [negative sentiment in the eight largest policy categories](outputs/figures/03_sentiment_by_topic.png), and [all three labels within agriculture by party and election](outputs/figures/04_topic_sentiment_703.png).

## What the reference folder is for

`reference/` is a saved snapshot of an earlier run. It is useful for checking a rebuild or showing results before running the models. The pipeline reads its inputs from `data/` and writes the current results to `data/processed/` and `outputs/`.

| Reference file | Why keep it |
|---|---|
| `master_passages.csv` | Check whether PDF extraction reproduced the same text and passage IDs. |
| `scores.csv` | Compare individual lexicon and RoBERTa labels against the saved run. |
| `by_party_year_and_legacy_policy.csv` | Look up the earlier party-year summary and keyword-based policy categories. |
| `by_policy_topic.csv` | Compare the earlier ManifestoBERTa topic summaries with your new tables. |
| `disagreement_examples.csv` | Inspect examples retained in the earlier run. The current audit is in `outputs/`. |
| `sentiment_confusion.csv` and `figures/01_sentiment_agreement.png` | Check the earlier label counts and preview the earlier agreement heatmap. |

## Main finding

In the completed Windows GPU run, the lexicon and RoBERTa assigned the same label to **6,201 of 9,572 passages (64.8%)**. RoBERTa assigned negative sentiment to 9.5% of passages, compared with 4.3% for the lexicon. The rebuilt master CSV matched the saved reference byte for byte, and both sentiment labels matched for every passage ID. Agreement measures consistency between the methods. An independent annotation study would be needed to estimate accuracy on these manifestos. [Results and limitations](REPORT.md)
