# -*- coding: utf-8 -*-
"""
Core ML code for Persian Spam Detection
نسخه ۲ (بهبودیافته) — دقت بالاتر، بدون نشت آستانه به داده تست

تغییرات نسبت به نسخه قبل:
  1. کلمات انگلیسی به‌جای حذف شدن، با حروف کوچک نگه داشته می‌شوند
  2. ویژگی‌های ساختاری جدید (تعداد خطوط، هدر فوروارد، لگاریتم طول و...)
  3. بردار واژگان بزرگ‌تر: کلمه (1,3) تا 15000 + کاراکتر (3,6) تا 15000
  4. آنسامبل جدید: LR + LinearSVC کالیبره + ComplementNB با وزن مساوی
  5. آستانه بر اساس بیشینه‌سازی F1 روی داده آموزش (نه تست)
"""

import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

RANDOM_STATE = 42

# ----------------------------------------------------------------
# تنظیمات TF-IDF
# ----------------------------------------------------------------
MAX_TFIDF_FEATURES = 15000
MAX_CHAR_FEATURES = 15000
MIN_DF = 2
MAX_DF = 0.95

# ----------------------------------------------------------------
# کلمات کلیدی اسپم فارسی
# ----------------------------------------------------------------
SPAM_KEYWORDS = [
    'کانال', 'تلگرام', 'لینک', 'بیو', 'فالو', 'لایک', 'سابسکرایب',
    'تخفیف', 'ویژه', 'خرید', 'فروش', 'قیمت', 'ارزان', 'رایگان',
    'کلیک', 'کنید', 'بزنید', 'برید', 'عضو', 'شدید', 'برنده',
    'جایزه', 'هدیه', 'بونوس', 'تومان', 'ریال', 'پرداخت',
    'درآمد', 'میلیونی', 'پولدار', 'سرمایه', 'سود', 'فروشگاه',
    'سفارش', 'ارسال', 'پست', 'تحویل', 'گارانتی', 'ضمانت',
    'محدود', 'فوری', 'آخرین', 'فرصت', 'تمام', 'شدنی',
    'ثبت', 'نام', 'عضویت', 'کد', 'تخفیف', 'off', 'sale',
    'whatsapp', 'telegram', 'instagram', 'اینستاگرام', 'واتساپ',
    '091', '092', '093', '090', '099',
]

NORMAL_KEYWORDS = [
    'سلام', 'درود', 'ممنون', 'متشکرم', 'مرسی', 'خوبی', 'خوبم',
    'چطوری', 'حالت', 'لطفا', 'خواهش', 'ببخشید', 'ببخش',
    'خدا', 'انشاءال', 'موفق', 'موفقیت', 'شاد', 'سلامت',
    'دوست', 'رفیق', 'داداش', 'آقا', 'خانم', 'استاد',
    'درس', 'دانشگاه', 'کنکور', 'ارشد', 'دکتری', 'تحصیل',
    'کتاب', 'مقاله', 'تحقیق', 'پروژه', 'تمرین', 'امتحان',
]


def normalize_persian(text):
    text = str(text)
    text = text.lstrip("\ufeff").replace("\ufeff", "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = re.compile(r"[\u0640ـ]+").sub("", text)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def desensor_text(text):
    CENSOR_RE = re.compile(r"([\u0600-\u06FFA-Za-z])[*٭+#~_.\-•]+(?=[\u0600-\u06FFA-Za-z])")
    prev = None
    while prev != text:
        prev = text
        text = CENSOR_RE.sub(r"\1", text)
    return text


def clean_text(text, **kwargs):
    text = str(text)
    text = normalize_persian(text)
    text = desensor_text(text)

    # جایگزینی entity‌ها
    text = re.sub(r"(https?://\S+|www\.\S+)", " URLTOKEN ", text)
    text = re.sub(r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}", " EMAILTOKEN ", text)
    text = re.sub(r"(?:\+?\d{1,3}[\s\-]?)?(?:09\d{9}|0\d{9,10})", " PHONETOKEN ", text)
    text = re.sub(r"@\w+", " MENTIONTOKEN ", text)
    text = re.sub(r"#(\w+)", r" HASHTAGTOKEN \1", text)

    # بهبود اصلی: کلمات انگلیسی حفظ می‌شوند (با حروف کوچک)
    text = text.lower()

    # تکرار حروف
    had_repeat = bool(re.search(r"(.)\1{2,}", text))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if had_repeat:
        text += " REPEATEDCHARTOKEN"

    # حذف ایموجی
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]", " ", text)

    # حذف نقطه‌گذاری
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


CLEAN_KWARGS = {}


# ----------------------------------------------------------------
# ویژگی‌های مهندسی‌شده (پایه + ساختاری جدید)
# ----------------------------------------------------------------
def extract_engineered_features(raw_text):
    text = str(raw_text)
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # شمارش‌ها
    n_urls = len(re.findall(r"(https?://\S+|www\.\S+)", text))
    n_emails = len(re.findall(r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}", text))
    n_phones = len(re.findall(r"(?:09\d{9}|0\d{9,10})", text))
    n_digits = len(re.findall(r"\d", text)) + len(re.findall(r"[۰-۹٠-٩]", text))
    n_excl = text.count("!") + text.count("؟!")
    n_quest = text.count("?") + text.count("؟")
    n_hashtags = len(re.findall(r"#\w+", text))
    n_mentions = len(re.findall(r"@\w+", text))
    n_en_chars = len(re.findall(r"[a-zA-Z]", text))
    n_fa_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    n_repeats = len(re.findall(r"(.)\1{2,}", text))
    n_special = len(re.findall(r"[!@#$%^&*()_+=\[\]{};':\"\\|,.<>/?]", text))

    # کلمات کلیدی اسپم
    text_lower = text.lower()
    spam_kw_count = sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)
    normal_kw_count = sum(1 for kw in NORMAL_KEYWORDS if kw in text_lower)

    # نسبت‌ها
    digit_ratio = n_digits / n_chars
    en_ratio = n_en_chars / n_chars
    fa_ratio = n_fa_chars / n_chars
    special_ratio = n_special / n_chars

    # ویژگی‌های ترکیبی
    has_link = 1 if (n_urls > 0 or 'لینک' in text_lower or 'link' in text_lower) else 0
    has_phone = 1 if n_phones > 0 else 0
    has_money = 1 if any(w in text_lower for w in ['تومان', 'ریال', 'قیمت', 'خرید', 'فروش']) else 0
    has_action = 1 if any(w in text_lower for w in ['کلیک', 'کنید', 'بزنید', 'برید', 'فالو', 'لایک']) else 0
    has_spam_kw = 1 if spam_kw_count >= 2 else 0

    # ویژگی‌های ساختاری جدید (نسخه ۲)
    n_lines = text.count("\n")
    has_fwd_header = 1 if re.search(r"\bTo:|\bFrom:|\bSubject:", text) else 0
    has_gmail_header = 1 if text.lstrip().startswith(("Gmail", "gmail")) else 0
    log_len = np.log1p(len(text))

    return [
        len(text),
        n_words,
        avg_word_len,
        n_digits,
        digit_ratio,
        n_urls,
        n_emails,
        n_phones,
        n_excl,
        n_quest,
        n_hashtags,
        n_mentions,
        en_ratio,
        fa_ratio,
        special_ratio,
        n_repeats,
        n_special,
        spam_kw_count,
        normal_kw_count,
        has_link,
        has_phone,
        has_money,
        has_action,
        has_spam_kw,
        n_lines,
        has_fwd_header,
        has_gmail_header,
        log_len,
    ]


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        return np.array([extract_engineered_features(t) for t in texts], dtype=float)


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **clean_kwargs):
        self.clean_kwargs = clean_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.tolist() if hasattr(X, "tolist") else list(X)
        return [clean_text(t, **self.clean_kwargs) for t in texts]


# ----------------------------------------------------------------
# Vectorizer
# ----------------------------------------------------------------
def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=MIN_DF,
            max_df=MAX_DF,
            sublinear_tf=True,
            max_features=MAX_TFIDF_FEATURES,
            strip_accents='unicode',
            analyzer='word',
        )),
        ("char_wb", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=MIN_DF,
            sublinear_tf=True,
            max_features=MAX_CHAR_FEATURES,
        )),
    ])


def build_pipeline(classifier):
    text_branch = Pipeline([
        ("clean", CleanTextTransformer(**CLEAN_KWARGS)),
        ("tfidf", build_vectorizer()),
    ])
    features = ColumnTransformer(transformers=[
        ("tfidf_branch", text_branch, "text"),
        ("engineered", Pipeline([
            ("extract", EngineeredFeatures()),
            ("scale", MaxAbsScaler()),
        ]), "text"),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def as_model_input(messages):
    return pd.DataFrame({"text": list(messages)})


# ----------------------------------------------------------------
# مدل‌ها
# ----------------------------------------------------------------
MODEL_NAME = "Ensemble v2 (LR + SVM + NB)"
MODEL_FILENAME = "ensemble_model.joblib"


def get_lr():
    return LogisticRegression(
        max_iter=3000, C=1.0, penalty="l2", solver="liblinear",
        class_weight='balanced', random_state=RANDOM_STATE,
    )


def get_svm():
    base = LinearSVC(
        max_iter=3000, C=0.5, class_weight='balanced', random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(base, cv=3)


def get_nb():
    return ComplementNB(alpha=0.5)


def get_rf():
    # برای سازگاری با کدهای قدیمی نگه داشته شده؛ در آنسامبل جدید استفاده نمی‌شود
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    )


def get_classifier():
    """
    آنسامبل نسخه ۲: سه مدل متنوع با وزن مساوی.
    انتخاب‌شده با 5-fold CV روی داده آموزش.
    """
    return VotingClassifier(
        estimators=[
            ('lr', get_lr()),
            ('svm', get_svm()),
            ('nb', get_nb()),
        ],
        voting='soft',
        weights=[1, 1, 1],
    )


def get_single_classifier():
    """مدل تکی برای cross-validation سریع‌تر"""
    return get_lr()


def evaluate_with_cv(X, y, cv=5):
    from sklearn.model_selection import cross_val_score
    clf = get_single_classifier()
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


def find_optimal_threshold_from_proba(proba, y):
    """
    آستانه بهینه بر اساس بیشینه‌سازی F1.
    باید روی احتمالات out-of-fold داده «آموزش» صدا زده شود تا نشتی به تست نباشد.
    """
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.25, 0.75, 0.01):
        pred = (proba >= threshold).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


def find_optimal_threshold(pipe, X_val, y_val):
    """سازگار با امضای قدیمی؛ برای استفاده صحیح ترجیحاً از
    find_optimal_threshold_from_proba با احتمالات OOF استفاده کنید."""
    proba = pipe.predict_proba(X_val)[:, 1]
    return find_optimal_threshold_from_proba(proba, y_val)
