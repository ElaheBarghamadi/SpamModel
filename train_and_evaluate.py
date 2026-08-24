# -*- coding: utf-8 -*-
"""
آموزش مدل Logistic Regression روی دیتاست تمیزشده و نمایش/ذخیره نمرات ارزیابی.

اجرا (از ریشه‌ی پروژه‌ی جنگو):
    python train_and_evaluate.py --data data/emails.csv

خروجی‌ها:
    - نمرات ارزیابی (Accuracy, Precision, Recall, F1, ROC-AUC) روی داده تست
    - ماتریس درهم‌ریختگی (Confusion Matrix)
    - فایل مدل آموزش‌دیده (.joblib)
    - فایل metadata.json شامل نمرات
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# core.py باید در دسترس باشه
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from detector.ml import core
except ImportError:
    import core


def main():
    parser = argparse.ArgumentParser(description="آموزش و ارزیابی مدل تشخیص اسپم")
    parser.add_argument("--data", "-d", default="data/emails.csv", help="مسیر فایل CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="نسبت داده تست (پیش‌فرض 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="seed برای تکرارپذیری")
    parser.add_argument("--save-dir", default="saved_models", help="پوشه‌ی ذخیره مدل و متادیتا")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"خطا: فایل داده پیدا نشد: {args.data}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        print("خطا: فایل باید ستون‌های 'text' و 'label' داشته باشه.", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    # تبدیل label به 0/1
    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print("=" * 55)
    print("۱) بارگذاری داده")
    print("=" * 55)
    print(f"تعداد کل نمونه‌ها: {len(df)}")
    print(f"توزیع برچسب‌ها: {df['label'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["label"]
    )
    X_train = core.as_model_input(train_df["text"])
    y_train = train_df["label"].values
    X_test = core.as_model_input(test_df["text"])
    y_test = test_df["label"].values

    print(f"\nآموزش: {len(train_df)} نمونه | تست: {len(test_df)} نمونه")

    print("\n" + "=" * 55)
    print(f"۲) آموزش مدل: {core.MODEL_NAME}")
    print("=" * 55)
    t0 = time.time()
    clf = core.get_classifier()
    pipe = core.build_pipeline(clf)
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"زمان آموزش: {train_time:.1f} ثانیه")

    print("\n" + "=" * 55)
    print("۳) ارزیابی روی داده تست")
    print("=" * 55)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test)
    rec = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test).tolist()

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix (سطر=واقعی، ستون=پیش‌بینی):")
    print(f"                 پیش‌بینی: عادی   پیش‌بینی: اسپم")
    print(f"واقعی: عادی      {cm[0][0]:<15} {cm[0][1]}")
    print(f"واقعی: اسپم      {cm[1][0]:<15} {cm[1][1]}")

    print("\nگزارش کامل sklearn:")
    print(classification_report(y_test, pred_test, target_names=["عادی (0)", "اسپم (1)"]))

    print("=" * 55)
    print("۴) ذخیره‌سازی")
    print("=" * 55)
    save_path = os.path.join(args.save_dir, core.MODEL_FILENAME)
    joblib.dump({"pipeline": pipe, "threshold": 0.5}, save_path)
    print(f"مدل ذخیره شد: {save_path}")

    metadata = {
        "display_name": core.MODEL_NAME,
        "n_samples_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "train_time_seconds": round(train_time, 1),
    }
    metadata_path = os.path.join(args.save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"متادیتا/نمرات ذخیره شد: {metadata_path}")

    print("\n" + "=" * 55)
    print("✅ آموزش با موفقیت انجام شد!")
    print("=" * 55)


if __name__ == "__main__":
    main()
