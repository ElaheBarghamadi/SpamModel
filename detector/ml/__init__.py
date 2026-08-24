# -*- coding: utf-8 -*-
"""
Core ML code for the Persian Spam Detection Django app.

This module is imported both by:
  - train_models.py (offline training / re-training of the saved models)
  - detector/views.py (loading saved models and running predictions)
"""

import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 10000
MIN_DF = 8

# ----------------------------------------------------------------
# Persian text preprocessing
# ----------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:09\d{9}|0\d{9,10})")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
REPEAT_RE = re.compile(r"(.)\1{2,}")
TATWEEL_RE = re.compile(r"[\u0640ـ]+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

CENSOR_CHARS = r"*٭+#~_\.\-•"
CENSOR_RE = re.compile(rf"([\u0600-\u06FFA-Za-z])[{CENSOR_CHARS}]+(?=[\u0600-\u06FFA-Za-z])")

KNOWN_TOKENS = {
    "URLTOKEN", "EMAILTOKEN", "PHONETOKEN", "MENTIONTOKEN",
    "HASHTAGTOKEN", "REPEATEDCHARTOKEN", "ENGWORDTOKEN",
}
GENERIC_ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")
MULTI_ENGWORD_RE = re.compile(r"(?: ENGWORDTOKEN){2,}")


def normalize_persian(text):
    text = str(text)
    text = text.lstrip("\ufeff")
    text = text.replace("\ufeff", "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = TATWEEL_RE.sub("", text)
    text = text.translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def desensor_text(text):
    prev = None
    while prev != text:
        prev = text
        text = CENSOR_RE.sub(r"\1", text)
    return text


def replace_entities(text):
    text = URL_RE.sub(" URLTOKEN ", text)
    text = EMAIL_RE.sub(" EMAILTOKEN ", text)
    text = PHONE_RE.sub(" PHONETOKEN ", text)
    text = MENTION_RE.sub(" MENTIONTOKEN ", text)
    text = HASHTAG_RE.sub(r" HASHTAGTOKEN \1", text)
    return text


def replace_generic_english(text):
    def _sub(match):
        word = match.group(0)
        if word.upper() in KNOWN_TOKENS:
            return word
        return " ENGWORDTOKEN "
    text = GENERIC_ENGLISH_RE.sub(_sub, text)
    text = MULTI_ENGWORD_RE.sub(" ENGWORDTOKEN", text)
    return text


def reduce_repeats(text):
    had_repeat = bool(REPEAT_RE.search(text))
    text = REPEAT_RE.sub(r"\1\1", text)
    if had_repeat:
        text = text + " REPEATEDCHARTOKEN"
    return text


def strip_punctuation(text):
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


def strip_emojis(text):
    return EMOJI_RE.sub(" ", text)


def clean_text(text, remove_punct=True, remove_emoji=True, entities="replace",
                fix_repeats=True):
    text = str(text)
    text = normalize_persian(text)
    text = desensor_text(text)
    if entities == "replace":
        text = replace_entities(text)
        text = replace_generic_english(text)
    if fix_repeats:
        text = reduce_repeats(text)
    if remove_emoji:
        text = strip_emojis(text)
    if remove_punct:
        text = strip_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


CLEAN_KWARGS = dict(remove_punct=True, remove_emoji=True, entities="replace",
                     fix_repeats=True)

# ----------------------------------------------------------------
# Engineered (non TF-IDF) count features
# ----------------------------------------------------------------
ENGLISH_RE = re.compile(r"[a-zA-Z]")
UPPER_EN_RE = re.compile(r"[A-Z]")
PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
DIGIT_RE = re.compile(r"\d")


def extract_engineered_features(raw_text):
    text = str(raw_text)
    n_chars = max(len(text), 1)
    words = text.split()

    n_urls = len(URL_RE.findall(text))
    n_emails = len(EMAIL_RE.findall(text))
    n_digits = len(DIGIT_RE.findall(text)) + len(re.findall(r"[۰-۹٠-٩]", text))
    n_excl = text.count("!") + text.count("؟!")
    n_quest = text.count("?") + text.count("؟")
    n_emoji = len(EMOJI_RE.findall(text))
    n_hashtags = len(HASHTAG_RE.findall(text))
    n_mentions = len(MENTION_RE.findall(text))
    n_upper_en = len(UPPER_EN_RE.findall(text))
    n_en_chars = len(ENGLISH_RE.findall(text))
    n_fa_chars = len(PERSIAN_CHAR_RE.findall(text))
    n_repeats = len(REPEAT_RE.findall(text))

    return [
        len(text), len(words), n_digits, n_urls, n_emails, n_excl, n_quest,
        n_emoji, n_hashtags, n_mentions, n_upper_en,
        n_en_chars / n_chars, n_fa_chars / n_chars, n_repeats,
    ]


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        return np.array([extract_engineered_features(t) for t in texts])


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **clean_kwargs):
        self.clean_kwargs = clean_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        return [clean_text(t, **self.clean_kwargs) for t in texts]


# ----------------------------------------------------------------
# Vectorizer + full pipeline builder
# ----------------------------------------------------------------
def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=MIN_DF, max_df=0.9,
                                  sublinear_tf=True, max_features=MAX_TFIDF_FEATURES)),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=MIN_DF,
                                     sublinear_tf=True, max_features=MAX_TFIDF_FEATURES)),
    ])


def build_pipeline(classifier):
    """raw text (as a DataFrame with a 'text' column) -> prediction."""
    text_branch = Pipeline([
        ("clean", CleanTextTransformer(**CLEAN_KWARGS)),
        ("tfidf", build_vectorizer()),
    ])
    features = ColumnTransformer(transformers=[
        ("tfidf_branch", text_branch, "text"),
        ("engineered", Pipeline([
            ("extract", EngineeredFeatures()),
            ("scale", StandardScaler(with_mean=False)),
        ]), "text"),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def as_model_input(messages):
    """Every saved pipeline expects a DataFrame with a 'text' column."""
    return pd.DataFrame({"text": list(messages)})


# ----------------------------------------------------------------
# Single production model: Logistic Regression
# ----------------------------------------------------------------
MODEL_NAME = "Logistic Regression"
MODEL_FILENAME = "logistic_regression.joblib"


def get_classifier():
    return LogisticRegression(max_iter=2000, C=1.0, penalty="l2", solver="liblinear")
