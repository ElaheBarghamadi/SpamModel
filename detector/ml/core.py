# -*- coding: utf-8 -*-
"""
Core ML code for Persian Spam Detection
نسخه بهبود یافته - ضد اورفیت + دقت بالا
"""

import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42

# ----------------------------------------------------------------
# تنظیمات بهینه‌شده TF-IDF
# ----------------------------------------------------------------
# کاهش max_features و افزایش min_df برای جلوگیری از اورفیت
MAX_TFIDF_FEATURES = 8000
MIN_DF = 3
MAX_DF = 0.85

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
# ویژگی‌های مهندسی‌شده بیشتر
# ----------------------------------------------------------------
ENGLISH_RE = re.compile(r"[a-zA-Z]")
UPPER_EN_RE = re.compile(r"[A-Z]")
PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
DIGIT_RE = re.compile(r"\d")
SPECIAL_CHAR_RE = re.compile(r"[!@#$%^&*()_+=\[\]{};':\"\\|,.<>/?]")
LINK_RE = re.compile(r"(http|www|\.com|\.ir|\.org|\.net)", re.IGNORECASE)
MONEY_RE = re.compile(r"(تومان|ریال|قیمت|خرید|فروش|تخفیف|ارزان|رایگان)", re.IGNORECASE)
ACTION_RE = re.compile(r"(کلیک|کنید|بزنید|برید|بیاید|عضو|فالو|لایک|سابسکرایب)", re.IGNORECASE)


def extract_engineered_features(raw_text):
    text = str(raw_text)
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

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
    n_special = len(SPECIAL_CHAR_RE.findall(text))
    n_links = len(LINK_RE.findall(text))
    n_money = len(MONEY_RE.findall(text))
    n_action = len(ACTION_RE.findall(text))

    # نسبت‌ها
    digit_ratio = n_digits / n_chars
    en_ratio = n_en_chars / n_chars
    fa_ratio = n_fa_chars / n_chars
    special_ratio = n_special / n_chars
    upper_ratio = n_upper_en / max(n_en_chars, 1)

    return [
        len(text),              # طول متن
        n_words,                # تعداد کلمات
        avg_word_len,           # میانگین طول کلمات
        n_digits,               # تعداد ارقام
        digit_ratio,            # نسبت ارقام
        n_urls,                 # تعداد URL
        n_emails,               # تعداد ایمیل
        n_links,                # تعداد لینک‌ها
        n_excl,                 # تعداد !
        n_quest,                # تعداد ؟
        n_emoji,                # تعداد ایموجی
        n_hashtags,             # تعداد هشتگ
        n_mentions,             # تعداد منشن
        n_upper_en,             # حروف بزرگ انگلیسی
        en_ratio,               # نسبت انگلیسی
        fa_ratio,               # نسبت فارسی
        special_ratio,          # نسبت کاراکتر خاص
        upper_ratio,            # نسبت حروف بزرگ
        n_repeats,              # تکرار حروف
        n_money,                # کلمات مالی
        n_action,               # کلمات اکشن
        n_special,              # کاراکترهای خاص
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
# Vectorizer بهینه‌شده
# ----------------------------------------------------------------
def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=MIN_DF,
            max_df=MAX_DF,
            sublinear_tf=True,
            max_features=MAX_TFIDF_FEATURES,
            strip_accents='unicode',
        )),
        ("char_wb", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=MIN_DF,
            sublinear_tf=True,
            max_features=5000,  # کاهش برای جلوگیری از اورفیت
        )),
    ])


def build_pipeline(classifier):
    """ساخت pipeline کامل"""
    text_branch = Pipeline([
        ("clean", CleanTextTransformer(**CLEAN_KWARGS)),
        ("tfidf", build_vectorizer()),
    ])
    features = ColumnTransformer(transformers=[
        ("tfidf_branch", text_branch, "text"),
        ("engineered", Pipeline([
            ("extract", EngineeredFeatures()),
            ("scale", MaxAbsScaler()),  # بهتر از StandardScaler برای داده‌های پراکنده
        ]), "text"),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def as_model_input(messages):
    return pd.DataFrame({"text": list(messages)})


# ----------------------------------------------------------------
# مدل بهینه‌شده با ضد اورفیت
# ----------------------------------------------------------------
MODEL_NAME = "Logistic Regression (Optimized)"
MODEL_FILENAME = "logistic_regression.joblib"


def get_classifier():
    """
    Logistic Regression با تنظیمات ضد اورفیت:
    - C=0.5: regularization قوی‌تر (کمتر از 1.0)
    - penalty='l2': regularization L2
    - solver='liblinear': مناسب برای داده‌های کوچک/متوسط
    - class_weight='balanced': مدیریت عدم تعادل کلاس‌ها
    """
    return LogisticRegression(
        max_iter=2000,
        C=0.5,              # regularization قوی‌تر
        penalty="l2",
        solver="liblinear",
        class_weight='balanced',  # مدیریت عدم تعادل
        random_state=RANDOM_STATE,
    )


def get_calibrated_classifier():
    """
    مدل کالیبره‌شده برای احتمالات دقیق‌تر
    """
    base_clf = get_classifier()
    return CalibratedClassifierCV(base_clf, cv=3, method='sigmoid')


def evaluate_with_cv(X, y, cv=5):
    """
    ارزیابی مدل با cross-validation برای بررسی اورفیت
    """
    clf = get_classifier()
    pipe = build_pipeline(clf)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    scores = {
        'accuracy': cross_val_score(pipe, X, y, cv=skf, scoring='accuracy'),
        'precision': cross_val_score(pipe, X, y, cv=skf, scoring='precision'),
        'recall': cross_val_score(pipe, X, y, cv=skf, scoring='recall'),
        'f1': cross_val_score(pipe, X, y, cv=skf, scoring='f1'),
        'roc_auc': cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc'),
    }
    
    return {k: (v.mean(), v.std()) for k, v in scores.items()}
