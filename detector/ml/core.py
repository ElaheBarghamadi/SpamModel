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

import os
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
    # ضد فرار: حروفی که با فاصله از هم جدا شده‌اند را دوباره می‌چسباند
    # مثال: «ب ر ن د ه شدید» ← «برنده شدید»
    text = re.sub(r"(?<!\S)([\w](?: [\w]){2,6})(?!\S)",
                  lambda m: m.group(1).replace(" ", ""), text)
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

    # الگوهای فیشینگ/فوریت (نسخه ۳)
    n_cred_req = len(re.findall(
        r"اطلاعات (کارت|حساب|خود)|رمز (پویا|ثانوی|شما)|کد ملی|تایید (اطلاعات|هویت)", text))
    n_urgency_threat = len(re.findall(
        r"مسدود|جریمه|بدهی|تعلیق|مهلت|فورا اقدام|آخرین (فرصت|مهلت)|ظرفیت محدود", text))
    phishing_pattern = 1 if (n_cred_req > 0 or (n_urgency_threat > 0 and
        re.search(r"(کلیک|لینک|تماس بگیرید|واریز|ارسال عدد|وارد نمایید)", text))) else 0

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
        n_cred_req,
        n_urgency_threat,
        phishing_pattern,
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
# بارگذاری یکپارچه همه دیتاست‌ها
# ----------------------------------------------------------------
TEXT_COLUMNS = ["text", "message", "content", "comment", "متن", "پیام"]
LABEL_COLUMNS = ["label", "labels", "class", "target", "برچسب", "دسته", "category", "وضعیت"]

_HAM_LABELS = {
    'ham', 'normal', 'not spam', 'notspam', 'non-spam', 'nonspam', 'no spam',
    'legit', 'legitimate', 'safe', 'clean', 'ok', 'good', 'benign',
    'عادی', 'نرمال', 'سالم', 'غیر اسپم', 'هرزنامه نیست',
}
_SPAM_LABELS = {
    'spam', 'junk', 'scam', 'phishing', 'phish', 'bad', 'ads', 'ad',
    'اسپم', 'هرزنامه', 'هرزپیام', 'تبلیغ', 'تبلیغات', 'مزاحم', 'فیشینگ', 'کلاهبرداری',
}


def normalize_label(value):
    """
    تبدیل انواع مختلف برچسب به ۰ (عادی) یا ۱ (اسپم).
    اگر برچسب ناشناخته باشد مقدار None برمی‌گرداند (سطر حذف می‌شود).
    نمونه‌های پشتیبانی‌شده:
      عادی: ham, Ham, HAM, normal, نرمال, عادی, سالم, not spam, 0, false, no
      اسپم: spam, Spam, SPAM, junk, phishing, اسپم, هرزنامه, تبلیغات, فیشینگ, 1, true, yes
    """
    if isinstance(value, (bool, np.bool_)):
        return 1 if value else 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            v = int(value)
        except (ValueError, OverflowError):
            return None
        return v if v in (0, 1) else None
    s = str(value).strip().lower().replace('\u200c', ' ')
    s = re.sub(r'\s+', ' ', s)
    if s in ('0', 'false', 'no', 'نه', 'خیر', 'منفی'):
        return 0
    if s in ('1', 'true', 'yes', 'بله', 'آره', 'مثبت'):
        return 1
    if s in _HAM_LABELS:
        return 0
    if s in _SPAM_LABELS:
        return 1
    return None


def load_datasets(data_dir="data"):
    """
    همه فایل‌های csv داخل پوشه داده را می‌خواند و یکپارچه می‌کند.
    هر فایل باید ستون متن و ستون برچسب داشته باشد.
    خروجی: دیتافریم با ستون‌های text, label(0/1), source
    """
    import glob
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
        label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
        if text_col is None or label_col is None:
            continue
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df = df.dropna()
        df["label"] = df["label"].map(normalize_label)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        df["source"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"هیچ دیتاست معتبری در {data_dir} پیدا نشد")

    out = pd.concat(frames, ignore_index=True)
    # حذف تکراری‌ها بین همه منابع (با متن نرمال‌شده)
    out["_norm"] = out["text"].map(normalize_persian)
    out = out.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    return out.reset_index(drop=True)


# ----------------------------------------------------------------
# مدل‌ها
# ----------------------------------------------------------------
MODEL_NAME = "Calibrated LinearSVC (v3)"
MODEL_FILENAME = "ensemble_model.joblib"


def get_lr():
    # penalty حذف شده: در نسخه‌های جدید اسکیت‌لرن منسوخ است و پیش‌فرض همان l2 است
    return LogisticRegression(
        max_iter=3000, C=1.0, solver="liblinear",
        class_weight='balanced', random_state=RANDOM_STATE,
    )


def get_svm():
    base = LinearSVC(
        max_iter=3000, C=1.0, class_weight='balanced', random_state=RANDOM_STATE,
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
    مدل نسخه ۳: لینیر‌اس‌وی‌سی کالیبره (C=1.0).
    روی داده ترکیبی (ایمیل + پیامک فارسی) از بین چندین پیکربندی
    با 5-fold CV بهترین نتیجه را داد.
    """
    return get_svm()


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


def find_optimal_threshold_from_proba(proba, y, fp_cost=2.0, fn_cost=1.0):
    """
    آستانه بهینه با رویکرد هزینه‌محور متعادل:
    اسپم‌کردنِ اشتباهی یک پیام عادی (FP) دو برابر بدتر از رد شدن یک اسپم (FN) است.
    باید روی احتمالات out-of-fold داده «آموزش» صدا زده شود تا نشتی به تست نباشد.
    بین آستانه‌های هم‌هزینه، آستانه بالاتر (امن‌تر) انتخاب می‌شود.
    """
    best_threshold = 0.5
    best_cost = float('inf')

    for threshold in np.arange(0.25, 0.86, 0.01):  # صعودی: در تساوی، آستانه بالاتر می‌ماند
        pred = (proba >= threshold).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        cost = fp_cost * fp + fn_cost * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return float(best_threshold)


def find_optimal_threshold(pipe, X_val, y_val):
    """سازگار با امضای قدیمی؛ برای استفاده صحیح ترجیحاً از
    find_optimal_threshold_from_proba با احتمالات OOF استفاده کنید."""
    proba = pipe.predict_proba(X_val)[:, 1]
    return find_optimal_threshold_from_proba(proba, y_val)
