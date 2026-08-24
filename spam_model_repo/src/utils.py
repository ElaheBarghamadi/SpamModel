"""
utils.py
--------
Shared constants and helper functions used across the project
(data loading, path handling, label derivation).

Keeping these helpers in one place avoids duplicating logic in
train.py and predict.py, and makes it easy to extend the project
later (e.g. adding a config file, new data sources, etc.).
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Everything is resolved relative to the project root so the scripts work
# no matter where they are called from.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Raw PHICAD files (download from https://github.com/davardoust/PHICAD
# and place them in the data/ folder before running train.py).
PHICAD_PART1_PATH = os.path.join(DATA_DIR, "PHICAD-part1.csv")
PHICAD_PART2_PATH = os.path.join(DATA_DIR, "PHICAD-part2.csv")

# Small, hand-curated set of obfuscated/manipulated comments (dot/star/
# underscore-separated letters, elongated letters, censor symbols, plus
# benign "hard negative" examples). Ships with the project — see
# data/augmented_samples.csv and README.md for details.
AUGMENTED_SAMPLES_PATH = os.path.join(DATA_DIR, "augmented_samples.csv")

# How many times each augmented row is repeated when merged into the
# training set. PHICAD has ~300k rows, so a handful of extra rows would
# have almost no effect on TF-IDF/LinearSVC training; repeating them
# gives these specific patterns enough weight to actually be learned,
# without overwhelming the real dataset (60 rows x 15 = 900 rows, well
# under 1% of the full dataset).
AUGMENTATION_REPEAT = 15

# Where the trained artifacts are saved/loaded from.
MODEL_PATH = os.path.join(MODELS_DIR, "linear_svc_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
# Final classes used by this project.
LABELS = ["Normal", "Spam", "Hate"]

# Column names as they appear in the raw PHICAD csv files.
TEXT_COLUMN = "comment_normalized"
RAW_COLUMNS = ["comment_normalized", "hate", "spam", "obscene", "class"]


def derive_label(hate: int, spam: int, obscene: int) -> str:
    """
    Map PHICAD's three binary flags (hate, spam, obscene) to a single
    target label: Normal, Spam, or Hate.

    Design decision (documented in README.md):
        - PHICAD ships an "Obscenity" category. As instructed, obscene
          content is merged into "Hate" because obscene comments are a
          form of harmful/toxic content, not a separate top-level class
          for this base version.
        - Priority order when multiple flags are set: Hate > Spam > Normal.
          This means any comment flagged as hate OR obscene becomes "Hate",
          even if it is also flagged as spam (e.g. a comment that is both
          spam and obscene is treated as Hate, since obscenity is the more
          severe signal).
        - A comment with none of the flags set is "Normal" (PHICAD calls
          this "clean").

    We derive the label directly from the three binary columns rather
    than from PHICAD's own combined "class" string column, because a
    few rows in the raw data have flag combinations (e.g. hate=1,
    spam=1) that are not covered by the "class" column's vocabulary.
    Using the binary flags is more robust and covers every row.
    """
    if hate == 1 or obscene == 1:
        return "Hate"
    if spam == 1:
        return "Spam"
    return "Normal"


def load_phicad_dataset() -> pd.DataFrame:
    """
    Load and combine both PHICAD csv parts into a single DataFrame with
    two columns: 'text' and 'label'.

    PHICAD is distributed as two tab-separated csv files:
        - PHICAD-part1.csv has a header row.
        - PHICAD-part2.csv does NOT have a header row (same column order).

    Returns
    -------
    pd.DataFrame with columns ['text', 'label']
    """
    if not os.path.exists(PHICAD_PART1_PATH) or not os.path.exists(PHICAD_PART2_PATH):
        raise FileNotFoundError(
            "PHICAD csv files not found in the data/ folder.\n"
            f"Expected:\n  {PHICAD_PART1_PATH}\n  {PHICAD_PART2_PATH}\n"
            "Download them from https://github.com/davardoust/PHICAD "
            "and place both files inside the data/ folder."
        )

    # part1 has a header row already matching RAW_COLUMNS
    part1 = pd.read_csv(PHICAD_PART1_PATH, sep="\t", encoding="utf-8")

    # part2 has no header row, so column names must be supplied explicitly
    part2 = pd.read_csv(
        PHICAD_PART2_PATH,
        sep="\t",
        encoding="utf-8",
        header=None,
        names=RAW_COLUMNS,
    )

    raw_df = pd.concat([part1, part2], ignore_index=True)

    # Drop rows with missing/empty comment text — they carry no signal.
    raw_df = raw_df.dropna(subset=[TEXT_COLUMN])
    raw_df = raw_df[raw_df[TEXT_COLUMN].str.strip() != ""]

    # Derive the final 3-class label from the binary flags.
    raw_df["label"] = raw_df.apply(
        lambda row: derive_label(row["hate"], row["spam"], row["obscene"]),
        axis=1,
    )

    dataset = raw_df[[TEXT_COLUMN, "label"]].rename(columns={TEXT_COLUMN: "text"})
    dataset = dataset.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return dataset


def load_augmented_samples(repeat: int = AUGMENTATION_REPEAT) -> pd.DataFrame:
    """
    Load the small hand-curated set of obfuscated/manipulated comments
    from data/augmented_samples.csv (columns: text, label).

    Each row is repeated `repeat` times so it carries enough weight
    during training (see AUGMENTATION_REPEAT for reasoning). If the
    file doesn't exist, an empty DataFrame is returned so callers can
    still run without it.

    Returns
    -------
    pd.DataFrame with columns ['text', 'label']
    """
    if not os.path.exists(AUGMENTED_SAMPLES_PATH):
        return pd.DataFrame(columns=["text", "label"])

    samples = pd.read_csv(AUGMENTED_SAMPLES_PATH, encoding="utf-8")
    samples = samples.dropna(subset=["text", "label"])

    if repeat > 1:
        samples = pd.concat([samples] * repeat, ignore_index=True)

    return samples[["text", "label"]].reset_index(drop=True)


def load_combined_dataset(include_augmented: bool = True) -> pd.DataFrame:
    """
    Load the full training dataset: PHICAD plus (optionally) the
    hand-curated augmented samples.

    De-duplication is applied to the PHICAD portion only (see
    load_phicad_dataset) — the augmented rows are intentionally
    repeated, so they are appended afterwards rather than being
    deduplicated away.

    Returns
    -------
    pd.DataFrame with columns ['text', 'label']
    """
    phicad_dataset = load_phicad_dataset()

    if not include_augmented:
        return phicad_dataset

    augmented_dataset = load_augmented_samples()
    if augmented_dataset.empty:
        return phicad_dataset

    combined = pd.concat([phicad_dataset, augmented_dataset], ignore_index=True)
    return combined


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
