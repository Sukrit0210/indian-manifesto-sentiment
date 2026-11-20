# Sentiment in Indian Election Manifestos

This project extends my 2024 manifesto text-mining study to 21 English Lok Sabha manifestos from six parties. It compares the original word-count sentiment method with RoBERTa and uses ManifestoBERTa to study tone within policy topics. [Original project](https://github.com/Sukrit0210/indian-manifesto-text-mining) · [Project report](REPORT.md)

## Research Questions

1. How often do the original lexicon and RoBERTa assign the same sentiment to a passage?
2. How do their sentiment estimates vary by party, election, and policy topic?
3. Where do the two methods disagree, and what kinds of language appear in those passages?
4. Does the choice of sentiment method change how parties rank within the same issue and election? How sensitive are the labels to small text changes?

## Folder Structure

```text
data/raw/          Original English manifesto PDFs kept locally
data/processed/    Clean passage dataset and source counts
outputs/           Sentiment scores, topic summaries, and disagreement tables
outputs/figures/   Plots in PNG and SVG
outputs/sensitivity/ Party orderings, text perturbation results, and an OCR sensitivity plot
reference/         Saved results from an earlier run
src/               Numbered Python scripts
REPORT.md          Methods, results, and conclusions
requirements.txt   Python dependencies
```

Place the source PDFs in `data/raw/` under the filenames listed in the source inventory. The edition notes document source gaps and extraction choices. [Source inventory](data/sources.csv) · [Edition notes](data/edition_notes.md)

## Scripts

Install 64-bit Python 3.14 and open PowerShell in the project folder. Run the full project with:

```powershell
py -3.14 run_all.py
```

The launcher creates `.venv314/`, installs the pinned dependencies and PyTorch 2.10.0 with CUDA 12.6, and checks that the GPU is available. The first run needs internet access for the packages and pinned model files. For PDFs stored elsewhere, use `py -3.14 run_all.py --pdf-dir "PATH_TO_PDFS"`.

After setup, select `.venv314\Scripts\python.exe` in VS Code and run the numbered scripts in the order below. To regenerate the analysis and figures from existing scores and topics, use `py -3.14 run_all.py --start 4`.

The launcher runs stages 1–7. With stage 6 outputs already present, run `py -3.14 run_all.py --start 7` for the expanded audit. Add `--no-third-model` to skip BART, which downloads its checkpoint on first use. Use `--start 6 --stop 6` to regenerate just the original perturbation and ordering tables.

## What Each Script Does

### 1. Build the Dataset

[01_build_dataset.py](src/01_build_dataset.py) extracts and cleans text from the PDFs, using English OCR where needed. It saves paragraphs and list items with their original page numbers and source links. [Master passages](data/processed/master_passages.csv) · [Source counts](data/processed/source_counts.csv)

### 2. Score Sentiment

[02_score_sentiment.py](src/02_score_sentiment.py) applies the original positive and negative word lists and runs pretrained RoBERTa in PyTorch. Each method assigns a positive, neutral, or negative label to the same passage. [Passage scores](outputs/scores.csv)

### 3. Assign Policy Topics

[03_assign_topics.py](src/03_assign_topics.py) assigns one of ManifestoBERTa’s 56 categories to each short text unit. It combines these assignments into topic weights for each passage. [Passage topics](outputs/passage_topics.csv) · [Model versions](data/models.json)

### 4. Analyze the Results

[04_analyze.py](src/04_analyze.py) compares the two sentiment methods and summarizes sentiment by party, election, and policy topic. The added disagreement audit flags negation, criticism, and passages that combine commitment and problem language. The examples retain their source links.

| Output | Contents |
|---|---|
| [Party and year](outputs/by_party_year.csv) · [Party, year and topic](outputs/by_party_year_topic.csv) · [Topic totals](outputs/by_topic.csv) | Sentiment shares, counts, and agreement |
| [Topic coverage](outputs/topic_coverage.csv) | Passages that received topic assignments |
| [Agreement counts](outputs/sentiment_confusion.csv) · [Comparison examples](outputs/comparison_examples.csv) | Counts and examples for combinations of the two labels |
| [Disagreements by cue](outputs/disagreement_by_cue.csv) · [All disagreements](outputs/sentiment_disagreements.csv) · [Examples by cue](outputs/disagreement_cue_examples.csv) | Audit counts and source-linked text |

### 5. Make the Plots

[05_plot.py](src/05_plot.py) saves four figures. The added agriculture plot shows all three sentiment shares for both methods, grouped by party and election.

- [Sentiment agreement](outputs/figures/01_sentiment_agreement.png)
- [Party and election trends](outputs/figures/02_party_year_trends.png)
- [Negative sentiment by policy topic](outputs/figures/03_sentiment_by_topic.png)
- [Sentiment within agriculture by party and election](outputs/figures/04_topic_sentiment_703.png)

### 6. Check Sentiment Sensitivity

[06_audit_sensitivity.py](src/06_audit_sensitivity.py) checks whether the two methods reverse a pair of parties’ ordering by negative sentiment within the same topic and election. It also scores source passages after swapping BJP and Congress names or simulating OCR errors, and tests four constructed negation examples. [Party ordering summary](outputs/sensitivity/ordering_summary.csv) · [Text change results](outputs/sensitivity/perturbation_summary.csv) · [OCR sensitivity plot](outputs/sensitivity/ocr_sensitivity.png)

### 7. Run the Expanded Audit

[07_expanded_audit.py](src/07_expanded_audit.py) compares two BART prompts on the same 1,050 passages and measures score direction on 105 negation edits. It also reports perturbation results by manifesto, party, and election and measures how far the methods’ party sentiment gaps differ. [BART agreement](outputs/sensitivity/expanded/bart_agreement.csv) · [Negation coverage](outputs/sensitivity/expanded/negation_coverage.csv) · [Perturbations by group](outputs/sensitivity/expanded/perturbations_by_group.csv) · [Gap sizes](outputs/sensitivity/expanded/ordering_gap_sizes.csv)

## Dataset

The dataset contains **9,572 passages**. Each row is a paragraph or list item. Passage totals vary by manifesto, so the party-year figures show the actual counts alongside the percentages.

| Party | Elections included |
|---|---|
| BJP, Congress, CPI(M) | 2009, 2014, 2019, 2024 |
| AIADMK | 2009, 2014, 2019 |
| DMK, TMC | 2014, 2019, 2024 |

Important columns include `passage_id`, `document_id`, `party`, `year`, `page`, `text`, `word_count`, `extraction`, and `source_url`. Use `passage_id` to join the master dataset with the sentiment scores and topic assignments.

The [reference folder](reference/) holds an earlier copy of the master dataset, the scores, summary tables, examples, and a plot. It is useful for checking a rerun. The completed GPU run reproduced the saved master CSV byte for byte and matched both saved sentiment labels for every passage.

## Main Results

| Label | Original lexicon | RoBERTa |
|---|---:|---:|
| Positive | 2,185 (22.8%) | 2,293 (24.0%) |
| Neutral | 6,979 (72.9%) | 6,366 (66.5%) |
| Negative | 408 (4.3%) | 913 (9.5%) |

The methods agreed on **6,201 of 9,572 passages (64.8%)**. ManifestoBERTa made **15,376** short-unit topic assignments across **9,493** passages. The disagreement audit found agreement was lower among passages containing the tracked negation cue than in the corpus overall. [Agreement table](outputs/sentiment_confusion.csv) · [Audit table](outputs/disagreement_by_cue.csv)

With at least 10 weighted passages for each party, **145 of 415** comparisons (34.9%) reversed across the two methods. The BJP–Congress name swap changed RoBERTa’s label on **9 of 296** passages and changed zero lexicon labels. At a 5% corruption probability per eligible character, simulated OCR changed **17 of 340** RoBERTa labels and **16 of 340** lexicon labels. [Party orderings](outputs/sensitivity/ordering_summary.csv) · [Perturbation counts](outputs/sensitivity/perturbation_summary.csv)

On the same **1,050 passages**, rewriting BART’s neutral hypothesis changed its label on **867 passages**. Neutral labels rose from **20 to 887**, and BART’s agreement with RoBERTa rose from **34.2% to 73.7%**. RoBERTa’s sentiment score moved in the expected direction on all **105 negation pairs**, comprising three direct source edits and 102 constructed clauses added to manifesto passages. The lexicon’s score stayed the same on every pair. [BART labels](outputs/sensitivity/expanded/bart_label_counts.csv) · [Agreement](outputs/sensitivity/expanded/bart_agreement.csv) · [Negation results](outputs/sensitivity/expanded/negation_summary.csv)

## Limitations

- RoBERTa was trained on social media. Its performance on English Indian manifestos needs independent sentiment annotation.
- OCR and automatic policy coding can introduce errors. Some party-years lack a verified English edition.
- Cue-based groups can overlap. A cue match provides an example to inspect, and some groups contain few passages.

## Conclusion

Both methods mostly assigned neutral sentiment. RoBERTa assigned negative sentiment more often, and the choice of method reversed about a third of party comparisons within the better-covered topics. These findings describe how the two methods measure tone in this corpus. Independent labels would be needed to judge their accuracy. [Full analysis](REPORT.md)

## References

- [Pilot project and lexicon baseline](https://github.com/Sukrit0210/indian-manifesto-text-mining)
- [CardiffNLP Twitter-RoBERTa sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [ManifestoBERTa policy category model](https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1)
