# -*- coding: utf-8 -*-
"""
تمیزکاری و کوچک‌سازی هوشمند دیتاست تشخیص اسپم فارسی

اجرا (فقط تمیزکاری):
    python clean_data.py --input data/emails.csv --output data/emails_clean.csv

اجرا (تمیزکاری + کوچک‌سازی هوشمند تا زیر ۵۰۰۰ ردیف):
    python clean_data.py --input data/emails.csv --output data/emails_clean.csv --target-size 4800
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# همون نرمال‌سازی متنی که در detector/ml/core.py استفاده میشه
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
TATWEEL_RE = re.compile(r"[\u0640ـ]+")
CENSOR_CHARS = r"*٭+#~_\.\-•"
CENSOR_RE = re.compile(rf"([\u0600-\u06FFA-Za-z])[{CENSOR_CHARS}]+(?=[\u0600-\u06FFA-Za-z])")


def normalize_persian(text: str) -> str:
    text = str(text)
    text = text.lstrip("\ufeff").replace("\ufeff", "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = TATWEEL_RE.sub("", text)
    text = text.translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def desensor_text(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = CENSOR_RE.sub(r"\1", text)
    return text


LABEL_MAP = {
    "ham": 0, "normal": 0, "not spam": 0, "notspam": 0, "0": 0, 0: 0,
    "spam": 1, "1": 1, 1: 1,
}


def normalize_label(value):
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    try:
        num = float(key)
        if num in (0, 1):
            return int(num)
    except ValueError:
        pass
    return np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    report = {}
    start = len(df)
    report["ردیف‌های اولیه"] = start

    if "text" not in df.columns:
        raise ValueError("ستون 'text' در فایل ورودی پیدا نشد.")
    label_col = "label" if "label" in df.columns else ("spam" if "spam" in df.columns else None)
    if label_col is None:
        raise ValueError("ستون برچسب ('label' یا 'spam') در فایل ورودی پیدا نشد.")
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # 1) حذف ردیف‌های خالی
    df = df.dropna(subset=["text", "label"]).copy()
    report["حذف‌شده (متن/برچسب خالی)"] = start - len(df)

    # 2) نرمال‌سازی متن + رفع سانسور
    df["text"] = df["text"].astype(str).apply(normalize_persian).apply(desensor_text)

    # 3) حذف متن‌هایی که بعد از نرمال‌سازی خالی شدن
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    report["حذف‌شده (متن خالی پس از نرمال‌سازی)"] = before - len(df)

    # 4) اعتبارسنجی و یکسان‌سازی برچسب
    df["label"] = df["label"].apply(normalize_label)
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    report["حذف‌شده (برچسب نامعتبر)"] = before - len(df)

    # 5) حذف ردیف‌های کاملاً تکراری
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    report["حذف‌شده (تکراری کامل)"] = before - len(df)

    # 6) تشخیص و حذف تناقض برچسب
    label_counts = df.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    before = len(df)
    df = df[~df["text"].isin(conflicting_texts)].reset_index(drop=True)
    report["حذف‌شده (تناقض برچسب برای یک متن)"] = before - len(df)
    report["تعداد متن‌های متناقض یافت‌شده"] = len(conflicting_texts)

    report["ردیف‌های نهایی"] = len(df)
    report["نسبت اسپم (label=1)"] = int((df["label"] == 1).sum())
    report["نسبت عادی (label=0)"] = int((df["label"] == 0).sum())

    return df, report


def main():
    parser = argparse.ArgumentParser(description="تمیزکاری دیتاست تشخیص اسپم فارسی")
    parser.add_argument("--input", "-i", default="data/emails.csv", help="مسیر فایل CSV خام")
    parser.add_argument("--output", "-o", default="data/emails_clean.csv", help="مسیر فایل خروجی تمیز")
    parser.add_argument("--seed", type=int, default=42, help="seed برای تکرارپذیری")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"خطا: فایل ورودی پیدا نشد: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    clean_df, report = clean_dataframe(df)

    print("=" * 50)
    print("گزارش تمیزکاری دیتاست")
    print("=" * 50)
    for key, value in report.items():
        print(f"{key}: {value}")

    clean_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("-" * 50)
    print(f"فایل نهایی ذخیره شد در: {args.output} ({len(clean_df)} ردیف)")


if __name__ == "__main__":
    main()
