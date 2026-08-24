"""
predict.py
----------
Load the trained model + TF-IDF vectorizer and classify new Persian
comments as Normal, Spam, or Hate.

Usage:
    # Classify a single comment passed on the command line
    python src/predict.py --text "سلام دوستان، لطفا کانالمو دنبال کنید"

    # Classify multiple comments from a text file (one comment per line)
    python src/predict.py --file path/to/comments.txt
"""

import argparse

import joblib

from preprocess import clean_text
from utils import MODEL_PATH, VECTORIZER_PATH


def load_artifacts():
    """Load the trained LinearSVC model and TF-IDF vectorizer from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Trained model/vectorizer not found. Run 'python src/train.py' "
            "first to train and save the model."
        ) from error
    return model, vectorizer


def predict_texts(texts, model, vectorizer):
    """
    Predict labels for a list of raw (uncleaned) texts.

    Applies the exact same cleaning pipeline used during training before
    vectorizing, so training and inference stay consistent.
    """
    cleaned = [clean_text(text) for text in texts]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    return predictions


def read_lines(file_path):
    """Read non-empty lines from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Classify Persian comments as Normal, Spam, or Hate."
    )
    parser.add_argument("--text", type=str, help="A single comment to classify.")
    parser.add_argument(
        "--file", type=str, help="Path to a text file with one comment per line."
    )
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide either --text or --file.")

    model, vectorizer = load_artifacts()

    if args.text:
        texts = [args.text]
    else:
        texts = read_lines(args.file)

    predictions = predict_texts(texts, model, vectorizer)

    for text, label in zip(texts, predictions):
        print(f"[{label}] {text}")


if __name__ == "__main__":
    main()
