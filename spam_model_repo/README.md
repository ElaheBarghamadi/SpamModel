# Persian Harmful Comment Classifier (PHICAD) — Base Project (v1)

A clean, modular, beginner-friendly **classical Machine Learning** project
that classifies Persian (Farsi) Instagram comments into three categories:

- **Normal**
- **Spam**
- **Hate**

This is **Version 1 (base project)**. It intentionally favors a simple,
maintainable foundation over squeezing out the highest possible score —
see [Future Work](#future-work) for what's designed to be added next.

No deep learning, no transformers, no BERT — only classical ML
(TF-IDF + LinearSVC), as required.

---

## Dataset

This project uses **PHICAD** (Persian Harmful Instagram Comments Annotated
Dataset): https://github.com/davardoust/PHICAD

- ~300,000 real Instagram comments in Farsi.
- Each comment has three binary flags: `hate`, `spam`, `obscene`.
- Distributed as two tab-separated CSV files:
  - `PHICAD-part1.csv` — **has** a header row.
  - `PHICAD-part2.csv` — has **no** header row (same column order as part1).

**Setup:** download both CSV files from the PHICAD repository above and
place them in the `data/` folder:

```
project/data/PHICAD-part1.csv
project/data/PHICAD-part2.csv
```

### Label mapping (3 classes)

PHICAD provides three binary flags rather than the three target classes
directly, so `src/utils.py::derive_label()` converts them as follows:

```
if hate == 1 or obscene == 1:  -> "Hate"
elif spam == 1:                -> "Spam"
else:                           -> "Normal"
```

**Why obscene is merged into Hate:** obscene comments are a form of
harmful/toxic content, not a separate top-level class in this base
version, so they are merged into `Hate` as instructed. When a comment is
flagged as both `spam` and `obscene`, it is treated as `Hate`, since
obscenity is considered the more severe signal. Comments with none of the
three flags set (PHICAD calls these "clean") become `Normal`.

The label is derived directly from the three binary columns rather than
from PHICAD's own combined `class` text column, because a small number of
rows have flag combinations not covered by that column's vocabulary
(e.g. `hate=1, spam=1`). Using the raw flags guarantees every row gets a
label.

Resulting class sizes on the full PHICAD dataset (after de-duplication):

| Label  | Count   |
|--------|---------|
| Normal | 162,942 |
| Hate   | 126,871 |
| Spam   | 11,647  |

### Augmented samples (data/augmented_samples.csv)

PHICAD's comments are real Instagram data, so filter-evasion tricks
(letters split with dots/stars, self-censoring with `****`, stretched
letters) are underrepresented in it, even though the preprocessing
pipeline is built to handle them (see [Preprocessing](#preprocessing)).
To make sure the model actually sees these patterns during training —
not just the preprocessing code being *able* to normalize them —
`data/augmented_samples.csv` ships with 60 short, hand-written
Persian comments (20 per class):

- **Hate** — generic insults/slurs written in obfuscated form
  (`ک.ی.ر تو دهنت`, `لاشی.ی.ی هستی`, `برو گمشو ****`, etc.)
- **Spam** — obfuscated ads/promotion patterns (spaced-out phone
  numbers, `کانال تلگرام`, `لینک بیو`, discount offers)
- **Normal** — benign comments that also use elongation or decorative
  stars (`سلاااام خوبی؟`, `واقعا عکس قشنگی بود ***`) — these are
  "hard negatives" so the model learns that stretched letters or a
  stray `*` don't automatically mean Hate/Spam.

`load_combined_dataset()` in `src/utils.py` loads this file and repeats
each row `AUGMENTATION_REPEAT` times (15x by default → ~900 rows) before
merging it with PHICAD, so these specific patterns carry enough weight
during training without overwhelming the ~300k real comments. `train.py`
uses this combined dataset by default.

This file is small and meant as a **starting point** — add more rows
(same `text,label` format) to teach the model new patterns as you find
them; no code changes are needed.

---

## Preprocessing

Implemented in `src/preprocess.py`. Applied identically at both training
and prediction time (via `clean_text()`), so the model always sees text
cleaned the same way it was trained on. **Stop words are NOT removed.**

Steps (in order):

1. **Remove URLs** — `http(s)://...` and `www...` links.
2. **Remove usernames** — `@mentions`.
3. **Remove emojis** — emoji/pictograph Unicode ranges.
4. **Normalize Persian characters** — e.g. `ي → ی`, `ك → ک`, plus other
   Arabic/Persian look-alike variants (`ى → ی`, `ة → ه`, `ؤ → و`, `أ/إ/ٱ → ا`,
   `ﷲ → الله`), and removes invisible formatting marks (ZWNJ, RTL/LTR marks).
5. **Normalize digits** — Persian (`۰۱۲۳...`) and Arabic-Indic
   (`٠١٢٣...`) digits are converted to plain Latin digits, so phone
   numbers/prices are represented consistently regardless of script.
6. **Remove diacritics** — strips Arabic vowel marks (tashkeel/harakat).
7. **Reduce letter elongation** — collapses a letter repeated 3+ times
   in a row to one occurrence (e.g. `سلاااام → سلام`, `خییییلی → خیلی`).
   This is common in both spam ("emphasis") and emotionally charged
   comments. Only applies to letters, not digits, so phone numbers are
   left untouched.
8. **Merge filter-evasion / obfuscated words** — reassembles words that
   were deliberately split into single letters joined by separators to
   dodge keyword filters, e.g. `ک.ی.ر → کیر`, `ک_ی_ر → کیر`,
   `ک*ی*ر → کیر`, `ک ی ر → کیر`. Only matches isolated single-letter
   sequences, so ordinary sentences are left untouched.
9. **Mark fully-censored tokens** — a token made entirely of symbols
   with no letters (e.g. `****`, `####`) can't be reconstructed, but the
   fact that something was self-censored there is itself a useful
   signal, so it's replaced with a placeholder word (`سانسور`) instead
   of being silently deleted.
10. **Remove unnecessary punctuation/symbols** — keeps Persian letters,
    digits, whitespace, and basic sentence punctuation (`. ! ? ؟`) so the
    text stays readable.
11. **Remove extra spaces** — collapses repeated whitespace and trims.

**Why this matters:** toxic/spam comments frequently try to evade simple
keyword filters using tricks like splitting words with dots/stars or
self-censoring with asterisks. Steps 7–9 specifically target these
tricks so the TF-IDF vectorizer sees the intended word (or an explicit
"censored" signal) instead of meaningless fragments.

---

## Feature Extraction

`TfidfVectorizer` (scikit-learn) with:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)
```

---

## Model

`LinearSVC` (scikit-learn), trained on the TF-IDF features with
`random_state=42`.

---

## Data Split

Stratified train/test split:

- 80% train / 20% test
- `random_state=42`
- Stratified on the label so class proportions are preserved in both sets.

---

## Evaluation

`train.py` prints:

- Accuracy
- Precision, Recall, F1-score (macro-averaged)
- Full classification report (per class)
- Confusion matrix (also saved as `models/confusion_matrix.png`)

Example results from a full run on PHICAD + the augmented samples:

```
Accuracy:  0.8877
Precision (macro): 0.8984
Recall (macro):    0.8680
F1-score (macro):  0.8823

              precision    recall  f1-score   support

      Normal       0.89      0.91      0.90     32484
        Spam       0.92      0.83      0.87      2389
        Hate       0.89      0.86      0.87     25345
```

These numbers are from the base pipeline with no tuning — expected to
improve once the future-work items below are added.

> Note: PHICAD's comments are already fairly clean, so the de-obfuscation
> steps and the small augmented set don't move these particular benchmark
> numbers by much — their real value shows up on messier real-world/
> adversarial input, which is exactly what a moderation system encounters
> in production even if it's underrepresented in this dataset. Example:
>
> ```
> python predict.py --text "ک.ی.ر تو دهنت"                    ->  [Hate]
> python predict.py --text "برو گمشو ****"                     ->  [Hate]
> python predict.py --text "لاشی ی ی هستی"                     ->  [Hate]
> python predict.py --text "کانال تلگرام لینک بیو فالو کنید"   ->  [Spam]
> python predict.py --text "سلاااام خوبی؟ عکس قشنگی بود ***"   ->  [Normal]
> ```

---

## Project Structure

```
project/
│
├── data/
│   ├── augmented_samples.csv # Hand-curated obfuscated/manipulated examples (ships with the project)
│   ├── PHICAD-part1.csv      # Place PHICAD-part1.csv here (download separately)
│   └── PHICAD-part2.csv      # Place PHICAD-part2.csv here (download separately)
├── models/                   # Trained model + vectorizer are saved here
├── src/
│   ├── preprocess.py         # Text cleaning pipeline
│   ├── train.py              # Load data -> train -> evaluate -> save
│   ├── predict.py            # Load saved model -> classify new comments
│   └── utils.py              # Shared paths, constants, data loading, label logic
├── requirements.txt
└── README.md
```

---

## How to Train

```bash
pip install -r requirements.txt

# make sure data/PHICAD-part1.csv and data/PHICAD-part2.csv exist
cd src
python train.py
```

This will:

1. Load and clean the dataset.
2. Split it (80/20, stratified).
3. Fit the TF-IDF vectorizer and train the LinearSVC model.
4. Print evaluation metrics.
5. Save `models/linear_svc_model.joblib` and `models/tfidf_vectorizer.joblib`.

---

## How to Predict

```bash
cd src

# classify a single comment
python predict.py --text "سلام دوستان، لطفا کانالمو دنبال کنید"

# classify many comments from a file (one per line)
python predict.py --file path/to/comments.txt
```

Output format: `[label] original text`

---

## Changelog

- **v1.2** — Added `data/augmented_samples.csv`: 60 hand-curated obfuscated/
  manipulated comments (20 per class) merged into training via
  `load_combined_dataset()` in `utils.py`, so the model actually sees
  filter-evasion patterns during training instead of relying solely on
  the preprocessing pipeline to normalize them at inference time.
- **v1.1** — Extended `preprocess.py`: broader Persian/Arabic character
  normalization, digit normalization, diacritic removal, letter-elongation
  collapsing, filter-evasion word merging (dot/star/underscore/hyphen/space
  separated letters), and fully-censored token detection (`****` → placeholder).
- **v1.0** — Initial base project (TF-IDF + LinearSVC, core preprocessing).

## Future Work

This base project is intentionally simple and extensible. Natural next
steps for later versions:

- `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning
  (TF-IDF params, SVM `C`, etc.)
- Cross-validation instead of a single train/test split
- Feature selection (e.g. `chi2`, `SelectKBest`)
- Trying other classical classifiers (Logistic Regression, Naive Bayes,
  Random Forest) and comparing them
- Deeper error analysis (inspect misclassified examples per class)
- More visualization (class distribution plots, learning curves, per-class
  metric charts)
- Optional `hazm`-based Persian normalization/lemmatization for stronger
  preprocessing
- A small model comparison report across configurations

The modular structure (`preprocess.py`, `utils.py`, `train.py`,
`predict.py`) is designed so each of these can be added without rewriting
the existing pipeline.
