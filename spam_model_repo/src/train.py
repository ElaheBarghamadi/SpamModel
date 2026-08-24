"""
train.py
--------
Main training script for the PHICAD Persian text classification project.

Pipeline:
    1. Load the raw PHICAD dataset (data/ folder).
    2. Clean the text with the preprocessing pipeline.
    3. Split into train/test sets (stratified, 80/20, random_state=42).
    4. Vectorize text with TF-IDF.
    5. Train a LinearSVC classifier.
    6. Evaluate on the test set (accuracy, precision, recall, F1,
       classification report, confusion matrix).
    7. Save the trained model and vectorizer to models/ with joblib.

Run:
    python src/train.py
"""

import matplotlib
matplotlib.use("Agg")  # allows saving plots without a display (headless-safe)

import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocess import clean_text_series
from utils import (
    LABELS,
    MODEL_PATH,
    MODELS_DIR,
    VECTORIZER_PATH,
    ensure_dir,
    load_combined_dataset,
)

# ---------------------------------------------------------------------------
# Configuration (kept as simple constants for this base version; a future
# version could move these into a config file / CLI arguments).
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)


def load_and_prepare_data():
    """Load the dataset (PHICAD + augmented samples) and clean the text."""
    print("Loading PHICAD dataset (+ hand-curated augmented samples)...")
    dataset = load_combined_dataset(include_augmented=True)
    print(f"Loaded {len(dataset)} comments (PHICAD + augmentation).")
    print("Label distribution:")
    print(dataset["label"].value_counts())

    print("\nCleaning text...")
    dataset["clean_text"] = clean_text_series(dataset["text"])

    # Drop rows that became empty after cleaning.
    dataset = dataset[dataset["clean_text"].str.strip() != ""].reset_index(drop=True)

    return dataset


def split_data(dataset):
    """Stratified train/test split (80/20)."""
    x_train, x_test, y_train, y_test = train_test_split(
        dataset["clean_text"],
        dataset["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset["label"],
    )
    return x_train, x_test, y_train, y_test


def vectorize_text(x_train, x_test):
    """Fit a TF-IDF vectorizer on the training text and transform both sets."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return vectorizer, x_train_vec, x_test_vec


def train_model(x_train_vec, y_train):
    """Train a LinearSVC classifier on the TF-IDF features."""
    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(x_train_vec, y_train)
    return model


def evaluate_model(model, x_test_vec, y_test):
    """Print evaluation metrics and save a confusion matrix plot."""
    y_pred = model.predict(x_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print("Confusion Matrix:")
    print(cm)

    # Save a confusion matrix plot for a quick visual check.
    ensure_dir(MODELS_DIR)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - LinearSVC (TF-IDF)")
    fig.tight_layout()
    cm_path = f"{MODELS_DIR}/confusion_matrix.png"
    fig.savefig(cm_path)
    print(f"\nConfusion matrix plot saved to: {cm_path}")


def save_artifacts(model, vectorizer):
    """Save the trained model and TF-IDF vectorizer with joblib."""
    ensure_dir(MODELS_DIR)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to:      {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


def main():
    dataset = load_and_prepare_data()
    x_train, x_test, y_train, y_test = split_data(dataset)

    print("\nVectorizing text with TF-IDF...")
    vectorizer, x_train_vec, x_test_vec = vectorize_text(x_train, x_test)

    print("Training LinearSVC classifier...")
    model = train_model(x_train_vec, y_train)

    evaluate_model(model, x_test_vec, y_test)
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()
