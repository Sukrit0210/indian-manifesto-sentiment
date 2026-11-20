
# Sentiment in Indian Election Manifestos

## Changes from the original pilot

The pilot used a balanced sample of 832 passages from four parties’ 2024 manifestos. It used a word-count lexicon for sentiment and explored topics with LDA and NMF. The extension expands coverage to six parties and four elections while keeping the lexicon as the baseline for passage-level comparison. The larger corpus supports comparisons over time. Party-year shares use the observed passage counts for each manifesto, and the figures show those denominators.

Pretrained RoBERTa adds a contextual sentiment score for each passage. ManifestoBERTa supplies a shared set of policy categories for tracing tone within the same issue. The completed analysis adds a disagreement audit in `04_analyze.py` and a party-by-election agriculture chart in `05_plot.py`. [Original pilot](https://github.com/Sukrit0210/indian-manifesto-text-mining) · [Disagreement audit](src/04_analyze.py) · [Chart code](src/05_plot.py)

## Data

The corpus comprises 21 English Lok Sabha manifestos and 9,572 retained passages or list items. BJP, Congress, and CPI(M) have editions for 2009, 2014, 2019, and 2024. AIADMK has editions for 2009, 2014, and 2019. DMK and TMC have editions for 2014, 2019, and 2024. The study required an existing English edition and could not verify one for AIADMK in 2024. Each passage has a stable ID, party, election, PDF page, source URL, and extraction method. [Source inventory](data/sources.csv) · [Source counts](data/processed/source_counts.csv) · [Edition notes](data/edition_notes.md)

The completed Windows run used Python 3.14 and PyTorch 2.10.0 with CUDA 12.6 on an RTX 4060 Laptop GPU. The rebuilt master CSV matched the earlier reference byte for byte. The two sentiment labels also matched the reference for all 9,572 passage IDs. This verifies reproducibility of the corpus and label decisions for this rerun. [Current corpus](data/processed/master_passages.csv) · [Saved corpus](reference/master_passages.csv) · [Current scores](outputs/scores.csv) · [Saved scores](reference/scores.csv)

## Methods

1. **PDF extraction.** Verify each edition using its recorded SHA-256. Extract text with PyMuPDF and apply English OCR where needed. Remove repeated page furniture, selected contents pages, and duplicate passages within a document. Preserve the PDF page and extraction method.
2. **Pilot lexicon.** Count the 22 positive and 21 negative words from the original project. The score is `(positive count − negative count) / (positive count + negative count + 1)`. Values above 0.05 are positive, values below −0.05 are negative, and the rest are neutral. [Lexicon code](src/lexicon.py)
3. **Contextual sentiment.** Apply pretrained CardiffNLP Twitter-RoBERTa to the same passages. Split long passages into windows of up to 256 tokens and average the window probabilities by token count. Use the highest probability to assign the passage label. The model was trained for sentiment on English social media. [RoBERTa model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
4. **Policy topics.** Split each passage into shorter units and assign one of the 56 ManifestoBERTa policy categories to each unit. If three of four units in a passage belong to one category, that category receives 0.75 passage weight. Each passage with topic assignments contributes one unit of weight across all its assigned categories. In a category name such as *Agriculture and Farmers: Positive*, “Positive” belongs to the policy coding scheme. Sentiment comes from the lexicon or RoBERTa. [ManifestoBERTa model card](https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1) · [Checkpoint hashes](data/models.json)
5. **Disagreement audit.** Compare the two sentiment labels and flag passages with simple text cues for negation, criticism, or a combination of commitment and problem language. Count disagreements among all passages that meet each cue. Categories can overlap. Review the source-linked examples to see how the language is used in context. [Analysis script](src/04_analyze.py)

## Results

### Agreement and class distribution

Both methods scored the same 9,572 passages. They agreed on **6,201 (64.8%)** and disagreed on **3,371 (35.2%)**.

| Sentiment label | Pilot lexicon | RoBERTa |
|---|---:|---:|
| Positive | 2,185 (22.8%) | 2,293 (24.0%) |
| Neutral | 6,979 (72.9%) | 6,366 (66.5%) |
| Negative | 408 (4.3%) | 913 (9.5%) |

Neutral was the most frequent label for both methods. The two largest disagreement groups were 1,278 lexicon-neutral passages that RoBERTa labelled positive, and 1,092 lexicon-positive passages that RoBERTa labelled neutral. [Agreement counts](outputs/sentiment_confusion.csv)

![Lexicon and RoBERTa sentiment agreement](outputs/figures/01_sentiment_agreement.png)

### Party, election, and policy topic

ManifestoBERTa made **15,376 short-unit assignments** across **9,493 passages**. The 79 passages without topic assignments still count in the overall sentiment analysis. The party-year plots show passage counts and leave gaps where a manifesto is absent from the corpus. [Topic coverage](outputs/topic_coverage.csv) · [Party-year figure](outputs/figures/02_party_year_trends.png)

Among the eight categories shown in the topic figure, *Political Authority* had the largest gap in negative sentiment. RoBERTa assigned negative labels to **32.7%** of its weighted passage total, compared with **6.9%** for the lexicon. The methods were much closer in *Welfare State Expansion*, at **3.9%** and **3.9%** to one decimal place. [Topic totals](outputs/by_topic.csv) · [Negative sentiment by policy topic](outputs/figures/03_sentiment_by_topic.png)

The agriculture breakdown shows why topic-level comparisons matter. For BJP, the RoBERTa positive share across the entire manifesto went from **53.4% in 2019** to **51.4% in 2024**. Within *Agriculture and Farmers: Positive*, it went from **46.6%** to **49.9%**. The agriculture estimates use 53 and 61 distinct passages, with 50.6 and 54.9 weighted passages respectively. The plot shows all three sentiment labels for both methods and labels each party-year’s denominator. [Party-year counts](outputs/by_party_year.csv) · [Topic breakdown](outputs/by_party_year_topic.csv)

![Agriculture sentiment by party, election, and method](outputs/figures/04_topic_sentiment_703.png)

### Where the methods disagree

| Text cue | Passages | Disagreements | Share |
|---|---:|---:|---:|
| All passages | 9,572 | 3,371 | 35.2% |
| Negation marker | 811 | 392 | 48.3% |
| Criticism marker | 38 | 28 | 73.7% |
| Commitment and problem language | 51 | 37 | 72.5% |
| None of the tracked cues | 8,688 | 2,924 | 33.7% |

The criticism and commitment-plus-problem groups had high disagreement rates in this corpus and small denominators. The cues are phrase matches, so the rates describe association within the corpus. They cannot establish which method made the better decision. One source-linked example is passage `BJP-2009-p017-b008-i001` on page 17. It introduces a commitment to food security and eliminating hunger. The lexicon labelled it negative and RoBERTa labelled it positive. [Cue summary](outputs/disagreement_by_cue.csv) · [Source-linked examples](outputs/disagreement_cue_examples.csv)

## Limitations and conclusion

Both methods found mostly neutral language. The choice of method materially affected how often passages were labelled negative, and the direction of a party-year change could differ within a single issue. The topic breakdown and source-linked examples make these differences inspectable.

RoBERTa was pretrained on a different domain. Topic assignments and OCR can introduce errors, and available English editions vary by party and year. The passages come from the same set of documents, so the shares describe the included corpus. An independently annotated sample would allow a future study to estimate sentiment accuracy and check whether the cue patterns explain the disagreements.
