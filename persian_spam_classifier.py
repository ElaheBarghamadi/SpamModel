# -*- coding: utf-8 -*-
"""
================================================================================
 طبقه‌بند اسپم ایمیل فارسی با رگرسیون لجستیک
 Persian Email Spam Classifier — Logistic Regression
================================================================================

این اسکریپت به صورت کامل شامل مراحل زیر است:
  1) بارگذاری داده
  2) نرمال‌سازی عمیق و اختصاصی متن فارسی (تمرکز اصلی پروژه)
  3) پاک‌سازی داده (Data Cleaning)
  4) استخراج ویژگی با TF-IDF + کنترل وزن کلمات/توکن‌های مصنوعی
  5) آموزش مدل رگرسیون لجستیک با جست‌وجوی ابرپارامتر (GridSearchCV)
  6) ارزیابی کامل مدل (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
  7) نمایش مهم‌ترین کلمات اثرگذار در تصمیم مدل

نحوه اجرا:
    python persian_spam_classifier.py [مسیر فایل csv]
    (در صورت ندادن مسیر، به صورت پیش‌فرض به دنبال "emails.csv" در همان پوشه می‌گردد)

فایل csv باید حداقل دو ستون داشته باشد:
    text  -> متن ایمیل
    label -> برچسب کلاس (مثلاً spam / ham)
================================================================================
"""

import sys
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ==============================================================================
# بخش ۱: جداول نگاشت و الگوهای مورد نیاز نرمال‌سازی متن فارسی
# ==============================================================================

# نگاشت حروف عربی به معادل فارسی رایج آن‌ها
ARABIC_TO_PERSIAN_CHARS = {
    "\u064A": "\u06CC",  # ي (عربی) -> ی (فارسی)
    "\u0643": "\u06A9",  # ك (عربی) -> ک (فارسی)
    "\u0629": "\u0647",  # ة (تاء مربوطه) -> ه
    "\u0624": "\u0648",  # ؤ -> و
    "\u0621": "\u0621",  # ء (بدون تغییر)
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0622": "\u0622",  # آ (بدون تغییر)
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0649": "\u06CC",  # ى (الف مقصوره) -> ی
    "\u06C1": "\u0647",  # ہ (اردو) -> ه
    "\u06D2": "\u06CC",  # ے (اردو) -> ی
    "\u0698\u0698": "\u0698",  # ژژ نادر -> ژ
}

# نگاشت ارقام عربی و فارسی به ارقام انگلیسی(لاتین)
DIGIT_MAP = {
    # ارقام فارسی
    "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3", "\u06F4": "4",
    "\u06F5": "5", "\u06F6": "6", "\u06F7": "7", "\u06F8": "8", "\u06F9": "9",
    # ارقام عربی (هندی-عربی)
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
}

# اعراب و علائم صوتی عربی (فتحه، ضمه، کسره، تشدید، سکون، تنوین) + کشیده (تطویل)
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

# کاراکترهای کنترلی/نامرئی که باید حذف شوند (BOM، ZWSP، RTL/LTR mark و ...)
INVISIBLE_CHARS_PATTERN = re.compile(
    r"[\uFEFF\u200B\u200D\u200E\u200F\u202A-\u202E\u2060\xa0]"
)

ZWNJ = "\u200C"  # نیم‌فاصله
PERSIAN_LETTER = r"[\u0621-\u0629\u062A-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC]"

# پیشوندهای فعل که باید با نیم‌فاصله به فعل بچسبند
ZWNJ_PREFIX_PATTERN = re.compile(r"\b(می|نمی)\s+(?=\S)")
# پسوندهای رایج که باید با نیم‌فاصله به کلمه قبل بچسبند
ZWNJ_SUFFIX_PATTERN = re.compile(r"(?<=\S)\s+(ها|های|هایی|هایم|هایت|هایش|تر|ترین|ام|ات|اش|ای)\b")

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|ir|org|net|info)(?:/\S*)?)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)|(?<!\d)0\d{9,10}(?!\d)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
LATIN_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")
NON_ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\u0621-\u0629\u062A-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC"
    r"\u200C0-9A-Za-z_ \n]"
)
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n+")

# توکن‌های مصنوعی (Placeholder) — کلماتی که جای عناصر غیر متنی/غیرفارسی را می‌گیرند.
# طبق خواسته‌ی پروژه، وزن این توکن‌ها در بردار ویژگی به صورت دستی کنترل/کاهش داده می‌شود
# تا صرفِ وجود لینک/ایمیل/کلمهٔ انگلیسیِ خاص، تصمیم مدل را با وزن نامتناسب منحرف نکند.
PLACEHOLDER_TOKENS = {
    "url": "نشانی_وب",
    "email": "پست_الکترونیک",
    "phone": "شماره_تلفن",
    "latin": "کلمه_خارجی",
}
# ضریب کاهش وزن اعمال‌شده روی ستون‌های TF-IDF مربوط به توکن‌های بالا (بین ۰ تا ۱)
PLACEHOLDER_WEIGHT_FACTOR = 0.25


# ==============================================================================
# بخش ۲: تابع نرمال‌سازی عمیق متن فارسی
# ==============================================================================

def fix_zwnj(text: str) -> str:
    """اصلاح نیم‌فاصله: چسباندن پیشوند/پسوندهای رایج و حذف نیم‌فاصله‌های نابجا."""
    # حذف نیم‌فاصله‌ای که بین دو حرف فارسی نیست (یعنی کنار فاصله/عدد/لاتین/علامت افتاده)
    text = re.sub(r"(?<!" + PERSIAN_LETTER + r")" + ZWNJ, " ", text)
    text = re.sub(ZWNJ + r"(?!" + PERSIAN_LETTER + r")", " ", text)
    # فشرده‌سازی نیم‌فاصله‌های تکراری
    text = re.sub(ZWNJ + r"+", ZWNJ, text)
    # چسباندن پیشوند فعل مضارع/نفی با نیم‌فاصله: "می خواهم" -> "می‌خواهم"
    text = ZWNJ_PREFIX_PATTERN.sub(lambda m: m.group(1) + ZWNJ, text)
    # چسباندن پسوند جمع/تفضیلی با نیم‌فاصله: "کتاب ها" -> "کتاب‌ها"
    text = ZWNJ_SUFFIX_PATTERN.sub(lambda m: ZWNJ + m.group(1), text)
    return text


def normalize_persian_text(text: str) -> str:
    """
    نرمال‌سازی کامل و چندلایه‌ی متن فارسی. این تابع قلب پروژه است و هدفش
    یکدست‌سازی کامل نگارش متن قبل از هر گونه استخراج ویژگی است:

      - حذف کاراکترهای نامرئی/کنترلی و BOM
      - یکدست‌سازی یونیکد (NFKC)
      - خنثی‌سازی برچسب‌های HTML باقی‌مانده
      - شناسایی و جایگزینی URL / ایمیل / شماره‌تلفن با توکن‌های مصنوعیِ کم‌وزن
      - تبدیل حروف عربی به معادل فارسی (ي->ی, ك->ک, ة->ه, ...)
      - تبدیل ارقام عربی/فارسی به ارقام انگلیسی
      - حذف اعراب و کشیدگی (تطویل)
      - شناسایی کلمات لاتین/انگلیسیِ باقی‌مانده و تبدیل به یک توکن فارسیِ کم‌وزن
        (بدون این‌که وزن کلمهٔ خاص انگلیسی در مدل بالا برود)
      - اصلاح نیم‌فاصله (ZWNJ) برای پیشوندها/پسوندهای رایج فارسی
      - حذف کشیدگی حروف (مثلاً «سلاااام» -> «سلام»)
      - حذف کاراکترهای خاص/علائم نگارشی غیرضروری
      - یکدست‌سازی فاصله‌ها و خطوط خالی
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # ۱) حذف کاراکترهای نامرئی/کنترلی (BOM, ZWSP, RTL/LTR mark و ...)
    text = INVISIBLE_CHARS_PATTERN.sub("", text)

    # ۲) یکدست‌سازی یونیکد (اشکال ترکیبی/سازگار حروف را یکی می‌کند)
    text = unicodedata.normalize("NFKC", text)

    # ۳) حذف برچسب‌های HTML باقی‌مانده (در ایمیل‌های اسپم رایج است)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # ۴) کنترل URL ها -> جایگزینی با توکن مصنوعی کم‌وزن (قبل از تشخیص ایمیل چون بعضی URLها ایمیل مانند هستند)
    text = URL_PATTERN.sub(f" {PLACEHOLDER_TOKENS['url']} ", text)

    # ۵) کنترل آدرس ایمیل -> توکن مصنوعی کم‌وزن
    text = EMAIL_PATTERN.sub(f" {PLACEHOLDER_TOKENS['email']} ", text)

    # ۶) کنترل شماره تلفن -> توکن مصنوعی کم‌وزن (باید قبل از map ارقام فارسی/عربی انجام شود که فقط ارقام لاتین می‌گیرد و نیز بعد از map تکرار می‌شود)
    text = PHONE_PATTERN.sub(f" {PLACEHOLDER_TOKENS['phone']} ", text)

    # ۷) تبدیل حروف عربی به معادل فارسی
    for ar, fa in ARABIC_TO_PERSIAN_CHARS.items():
        text = text.replace(ar, fa)

    # ۸) تبدیل ارقام فارسی/عربی به ارقام انگلیسی
    for src, dst in DIGIT_MAP.items():
        text = text.replace(src, dst)

    # شماره تلفن ممکن است در متن اصلی با ارقام فارسی/عربی بوده باشد؛ بعد از تبدیل ارقام دوباره بررسی می‌کنیم
    text = PHONE_PATTERN.sub(f" {PLACEHOLDER_TOKENS['phone']} ", text)

    # ۹) حذف اعراب/تشدید/تنوین/کشیدگی(تطویل)
    text = DIACRITICS_PATTERN.sub("", text)

    # ۱۰) شناسایی کلمات لاتین باقی‌مانده (کلمات انگلیسی/فینگلیش) و تبدیل به یک توکن فارسیِ کم‌وزن.
    #     نکته‌ی مهم پروژه: به‌جای نگه‌داشتن خودِ کلمهٔ انگلیسی (که می‌تواند به یک ویژگیِ
    #     پرقدرت و با وزن بالا در مدل تبدیل شود)، همهٔ این کلمات به یک توکنِ عمومی
    #     نگاشت می‌شوند و وزن نهاییِ آن توکن در مرحلهٔ TF-IDF عمداً پایین نگه داشته می‌شود.
    text = LATIN_WORD_PATTERN.sub(f" {PLACEHOLDER_TOKENS['latin']} ", text)

    # ۱۱) اصلاح نیم‌فاصله
    text = fix_zwnj(text)

    # ۱۲) حذف کشیدگیِ حروف تکراری (سلاااام -> سلام)
    text = ELONGATION_PATTERN.sub(r"\1\1", text)

    # ۱۳) حذف کاراکترهای خاص/علائم نگارشی/نمادها که اطلاعات معنایی مفیدی برای مدل ندارند
    text = NON_ALLOWED_CHARS_PATTERN.sub(" ", text)

    # ۱۴) یکدست‌سازی فاصله‌ها و خطوط
    text = MULTI_NEWLINE_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = text.strip()

    return text


# ==============================================================================
# بخش ۳: پاک‌سازی داده (Data Cleaning) در سطح دیتافریم
# ==============================================================================

def load_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    print(f"[۱] در حال خواندن فایل: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"text", "label"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"فایل csv باید ستون‌های {required_cols} را داشته باشد. ستون‌های موجود: {list(df.columns)}")

    n_before = len(df)

    # حذف رکوردهای تکراری کامل
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # حذف رکوردهایی که متن یا برچسبشان خالی/نامعتبر است
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # یکدست‌سازی برچسب‌ها (حذف فاصله اضافه و کوچک‌سازی حروف در صورت لاتین بودن)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    print(f"    تعداد رکورد قبل از پاک‌سازی: {n_before} | بعد از حذف تکراری/خالی: {len(df)}")
    print(f"    توزیع کلاس‌ها:\n{df['label'].value_counts().to_string()}")

    # اعمال نرمال‌سازی عمیق فارسی روی تمام متون
    print("[۲] در حال نرمال‌سازی متن (این بخش اصلی‌ترین قسمت پیش‌پردازش است)...")
    df["clean_text"] = df["text"].apply(normalize_persian_text)

    # حذف متن‌هایی که بعد از پاک‌سازی خالی شده‌اند (مثلاً فقط لینک/کد بودند)
    before_empty_drop = len(df)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    if before_empty_drop != len(df):
        print(f"    {before_empty_drop - len(df)} رکورد که بعد از نرمال‌سازی خالی شدند حذف شدند.")

    # حذف متون بسیار کوتاه که سیگنال معناداری ندارند (کمتر از ۲ توکن)
    token_counts = df["clean_text"].str.split().apply(len)
    df = df[token_counts >= 2].reset_index(drop=True)

    print(f"    تعداد نهایی رکوردها پس از پاک‌سازی کامل: {len(df)}")
    return df


# ==============================================================================
# بخش ۴: کنترل وزن توکن‌های مصنوعی در ماتریس TF-IDF
# ==============================================================================

def downweight_placeholder_columns(matrix, feature_names, placeholder_words, factor):
    """
    وزن ستون‌های TF-IDF مربوط به توکن‌های مصنوعی (لینک/ایمیل/تلفن/کلمهٔ خارجی) را
    با یک ضریب ثابت کاهش می‌دهد تا این ویژگی‌ها -صرفِ نظر از فراوانی‌شان- نتوانند
    وزن نامتناسبی نسبت به کلمات معنادار فارسی در تصمیم مدل پیدا کنند.

    این تابع علاوه بر خودِ توکن‌های مصنوعی، n-gram هایی را هم که صرفاً از ترکیب
    همین توکن‌های مصنوعی ساخته شده‌اند (مثل "کلمه_خارجی کلمه_خارجی" یا
    "نشانی_وب کلمه_خارجی") شناسایی و کم‌وزن می‌کند تا bigram/trigram های ساختگی
    هم از قانون کنترل وزن فرار نکنند.
    """
    placeholder_set = set(placeholder_words)
    matrix = matrix.tolil()
    affected = []
    for idx, name in enumerate(feature_names):
        tokens = name.split(" ")
        if all(tok in placeholder_set for tok in tokens):
            matrix[:, idx] = matrix[:, idx] * factor
            affected.append(name)
    return matrix.tocsr(), affected


# ==============================================================================
# بخش ۵: بدنه‌ی اصلی برنامه — آموزش و ارزیابی مدل
# ==============================================================================

def main(csv_path: str = "emails.csv"):
    # ---------- بارگذاری و پاک‌سازی داده ----------
    df = load_and_clean_dataset(csv_path)

    # کدگذاری برچسب: spam=1 ، هر برچسب دیگر (مثلاً ham)=0
    positive_label_candidates = [l for l in df["label"].unique() if "spam" in l]
    positive_label = positive_label_candidates[0] if positive_label_candidates else sorted(df["label"].unique())[0]
    y = (df["label"] == positive_label).astype(int).values
    print(f"[۳] برچسب کلاس مثبت (اسپم) در نظر گرفته‌شده: '{positive_label}'")

    X_text = df["clean_text"].values

    # ---------- تقسیم داده به آموزش/آزمون ----------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[۴] تقسیم داده: {len(X_train_text)} نمونه آموزش | {len(X_test_text)} نمونه آزمون")

    # ---------- استخراج ویژگی TF-IDF ----------
    print("[۵] ساخت بردارهای TF-IDF (unigram + bigram)...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    feature_names = vectorizer.get_feature_names_out()
    print(f"    تعداد ویژگی‌های استخراج‌شده: {X_train_tfidf.shape[1]}")

    # ---------- کنترل وزن توکن‌های مصنوعی (URL/ایمیل/تلفن/کلمهٔ خارجی) ----------
    X_train_tfidf, affected_train = downweight_placeholder_columns(
        X_train_tfidf, feature_names, PLACEHOLDER_TOKENS.values(), PLACEHOLDER_WEIGHT_FACTOR
    )
    X_test_tfidf, _ = downweight_placeholder_columns(
        X_test_tfidf, feature_names, PLACEHOLDER_TOKENS.values(), PLACEHOLDER_WEIGHT_FACTOR
    )
    if affected_train:
        print(f"    وزن توکن‌های مصنوعی {affected_train} با ضریب {PLACEHOLDER_WEIGHT_FACTOR} کاهش یافت.")

    # ---------- جست‌وجوی ابرپارامتر برای رگرسیون لجستیک ----------
    print("[۶] جست‌وجوی بهترین ابرپارامترها (GridSearchCV, 5-fold)...")
    param_grid = [
        {"penalty": ["l2"], "solver": ["liblinear"], "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]},
        {"penalty": ["l1"], "solver": ["liblinear"], "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]},
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train_tfidf, y_train)
    best_model = grid.best_estimator_
    print(f"    بهترین ابرپارامترها: {grid.best_params_}")
    print(f"    بهترین امتیاز F1 در اعتبارسنجی متقاطع: {grid.best_score_:.4f}")

    # ---------- اعتبارسنجی متقاطع نهایی روی مدل منتخب (سنجش پایداری) ----------
    cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=cv, scoring="f1")
    print(f"[۷] امتیازهای F1 در ۵-فولد (مدل نهایی): {np.round(cv_scores, 4)}")
    print(f"    میانگین F1 اعتبارسنجی متقاطع: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # ---------- ارزیابی روی مجموعه آزمون (Hold-out Test Set) ----------
    y_pred = best_model.predict(X_test_tfidf)
    y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("نتایج نهایی روی مجموعه آزمون (Test Set)")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nConfusion Matrix ([[TN, FP], [FN, TP]]):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham(غیراسپم)", "spam(اسپم)"]))

    # ---------- نمایش مهم‌ترین کلمات اثرگذار در تصمیم مدل ----------
    print("=" * 60)
    print("۲۰ ویژگیِ با بیشترین اثر در تشخیص «اسپم»:")
    coefs = best_model.coef_[0]
    top_spam_idx = np.argsort(coefs)[-20:][::-1]
    for idx in top_spam_idx:
        marker = " (توکن مصنوعیِ کم‌وزن)" if feature_names[idx] in PLACEHOLDER_TOKENS.values() else ""
        print(f"    {feature_names[idx]:<25} وزن={coefs[idx]:.4f}{marker}")

    print("\n۲۰ ویژگیِ با بیشترین اثر در تشخیص «غیراسپم» (ham):")
    top_ham_idx = np.argsort(coefs)[:20]
    for idx in top_ham_idx:
        marker = " (توکن مصنوعیِ کم‌وزن)" if feature_names[idx] in PLACEHOLDER_TOKENS.values() else ""
        print(f"    {feature_names[idx]:<25} وزن={coefs[idx]:.4f}{marker}")

    print("=" * 60)
    print("پایان اجرا.")

    return {
        "vectorizer": vectorizer,
        "model": best_model,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc},
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "emails.csv"
    main(csv_arg)
