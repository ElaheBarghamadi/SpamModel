# -*- coding: utf-8 -*-
"""
آموزش مدل بهینه‌شده با ضد اورفیت و ارزیابی کامل

اجرا:
    python train_and_evaluate.py --data data/emails.csv
    python train_and_evaluate.py --data data/emails.csv --cv 5
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from detector.ml import core
except ImportError:
    import core


def main():
    parser = argparse.ArgumentParser(description="آموزش و ارزیابی مدل تشخیص اسپم")
    parser.add_argument("--data", "-d", default="data/emails.csv", help="مسیر فایل CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="نسبت داده تست")
    parser.add_argument("--cv", type=int, default=5, help="تعداد fold برای cross-validation")
    parser.add_argument("--random-state", type=int, default=42, help="seed")
    parser.add_argument("--save-dir", default="saved_models", help="پوشه ذخیره")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"خطا: فایل داده پیدا نشد: {args.data}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) بارگذاری و آماده‌سازی داده
    # ----------------------------------------------------------------
    print("=" * 60)
    print("۱) بارگذاری داده")
    print("=" * 60)

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        print("خطا: فایل باید ستون‌های 'text' و 'label' داشته باشه.", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"تعداد کل نمونه‌ها: {len(df)}")
    print(f"توزیع: عادی={int((df['label']==0).sum())}, اسپم={int((df['label']==1).sum())}")
    print(f"نسبت: {int((df['label']==1).sum())/len(df)*100:.1f}% اسپم")

    # ----------------------------------------------------------------
    # 2) تقسیم داده
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("۲) تقسیم داده")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["label"]
    )
    X_train = core.as_model_input(train_df["text"])
    y_train = train_df["label"].values
    X_test = core.as_model_input(test_df["text"])
    y_test = test_df["label"].values

    print(f"آموزش: {len(train_df)} نمونه")
    print(f"تست: {len(test_df)} نمونه")

    # ----------------------------------------------------------------
    # 3) آموزش مدل
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"۳) آموزش مدل: {core.MODEL_NAME}")
    print("=" * 60)

    t0 = time.time()
    clf = core.get_classifier()
    pipe = core.build_pipeline(clf)
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"زمان آموزش: {train_time:.1f} ثانیه")

    # ----------------------------------------------------------------
    # 4) ارزیابی روی داده تست
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("۴) ارزیابی روی داده تست")
    print("=" * 60)

    # احتمالات
    proba_train = pipe.predict_proba(X_train)[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    pred_train = (proba_train >= 0.5).astype(int)

    # نمرات تست
    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test)
    rec = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test).tolist()

    # نمرات آموزش (برای بررسی اورفیت)
    acc_train = accuracy_score(y_train, pred_train)

    print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-" * 55)
    print(f"{'Accuracy':<20} {acc_train:>10.4f} {acc:>10.4f} {acc_train-acc:>10.4f}")
    print(f"{'Precision':<20} {'':>10} {prec:>10.4f}")
    print(f"{'Recall':<20} {'':>10} {rec:>10.4f}")
    print(f"{'F1-score':<20} {'':>10} {f1:>10.4f}")
    print(f"{'ROC-AUC':<20} {'':>10} {roc_auc:>10.4f}")

    # بررسی اورفیت
    gap = acc_train - acc
    print(f"\n🔍 تحلیل اورفیت:")
    print(f"   اختلاف دقت آموزش-تست: {gap:.4f}")
    if gap < 0.02:
        print(f"   ✅ عالی! مدل اورفیت نشده (gap < 2%)")
    elif gap < 0.05:
        print(f"   ⚠️ قابل قبول (gap < 5%)")
    else:
        print(f"   ❌ اورفیت! (gap > 5%) - نیاز به regularization بیشتر")

    print(f"\nماتریس اغتشاش:")
    print(f"                 پیش‌بینی: عادی   پیش‌بینی: اسپم")
    print(f"واقعی: عادی      {cm[0][0]:<15} {cm[0][1]}")
    print(f"واقعی: اسپم      {cm[1][0]:<15} {cm[1][1]}")

    print(f"\nگزارش طبقه‌بندی:")
    print(classification_report(y_test, pred_test, target_names=["عادی (0)", "اسپم (1)"]))

    # ----------------------------------------------------------------
    # 5) Cross-Validation
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"۵) Cross-Validation ({args.cv}-Fold)")
    print("=" * 60)

    print("در حال ارزیابی...")
    cv_scores = core.evaluate_with_cv(
        core.as_model_input(df["text"]), df["label"].values, cv=args.cv
    )

    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 45)
    for metric, (mean, std) in cv_scores.items():
        print(f"{metric:<20} {mean:>10.4f} {std:>10.4f}")

    # بررسی پایداری
    print(f"\n🔍 تحلیل پایداری:")
    f1_mean, f1_std = cv_scores['f1']
    if f1_std < 0.02:
        print(f"   ✅ مدل پایدار (F1 std={f1_std:.4f})")
    elif f1_std < 0.05:
        print(f"   ⚠️ پایداری متوسط (F1 std={f1_std:.4f})")
    else:
        print(f"   ❌ مدل ناپایدار (F1 std={f1_std:.4f}) - نیاز به داده بیشتر")

    # ----------------------------------------------------------------
    # 6) ذخیره‌سازی
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("۶) ذخیره‌سازی")
    print("=" * 60)

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
        "train_accuracy": round(acc_train, 4),
        "overfit_gap": round(gap, 4),
        "confusion_matrix": cm,
        "train_time_seconds": round(train_time, 1),
        "cv_scores": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in cv_scores.items()},
    }
    metadata_path = os.path.join(args.save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"متادیتا ذخیره شد: {metadata_path}")

    print("\n" + "=" * 60)
    print("✅ آموزش با موفقیت انجام شد!")
    print("=" * 60)


if __name__ == "__main__":
    main()
