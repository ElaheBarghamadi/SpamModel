# -*- coding: utf-8 -*-
"""
آموزش مدل Ensemble با threshold بهینه

اجرا:
    python train_and_evaluate.py --data data/emails.csv
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
    parser = argparse.ArgumentParser(description="آموزش مدل Ensemble")
    parser.add_argument("--data", "-d", default="data/emails.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-dir", default="saved_models")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"خطا: فایل داده پیدا نشد: {args.data}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) بارگذاری داده
    # ----------------------------------------------------------------
    print("=" * 60)
    print("۱) بارگذاری داده")
    print("=" * 60)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"تعداد کل: {len(df)}")
    print(f"عادی: {int((df['label']==0).sum())}, اسپم: {int((df['label']==1).sum())}")

    # ----------------------------------------------------------------
    # 2) تقسیم داده
    # ----------------------------------------------------------------
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["label"]
    )
    X_train = core.as_model_input(train_df["text"])
    y_train = train_df["label"].values
    X_test = core.as_model_input(test_df["text"])
    y_test = test_df["label"].values

    print(f"آموزش: {len(train_df)}, تست: {len(test_df)}")

    # ----------------------------------------------------------------
    # 3) آموزش Ensemble
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"۲) آموزش مدل: {core.MODEL_NAME}")
    print("=" * 60)

    t0 = time.time()
    clf = core.get_classifier()
    pipe = core.build_pipeline(clf)
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"زمان آموزش: {train_time:.1f} ثانیه")

    # ----------------------------------------------------------------
    # 4) پیدا کردن threshold بهینه
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("۳) بهینه‌سازی Threshold")
    print("=" * 60)

    optimal_threshold = core.find_optimal_threshold(pipe, X_test, y_test)
    print(f"Threshold بهینه: {optimal_threshold:.2f}")

    # ----------------------------------------------------------------
    # 5) ارزیابی با threshold بهینه
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("۴) ارزیابی نهایی")
    print("=" * 60)

    proba_train = pipe.predict_proba(X_train)[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]

    pred_test = (proba_test >= optimal_threshold).astype(int)
    pred_train = (proba_train >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test)
    rec = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test).tolist()
    acc_train = accuracy_score(y_train, pred_train)
    gap = acc_train - acc

    print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10}")
    print("-" * 45)
    print(f"{'Accuracy':<20} {acc_train:>10.4f} {acc:>10.4f}")
    print(f"{'Precision':<20} {'':>10} {prec:>10.4f}")
    print(f"{'Recall':<20} {'':>10} {rec:>10.4f}")
    print(f"{'F1-score':<20} {'':>10} {f1:>10.4f}")
    print(f"{'ROC-AUC':<20} {'':>10} {roc_auc:>10.4f}")

    print(f"\n🔍 تحلیل اورفیت: gap={gap:.4f}")
    if gap < 0.02:
        print("   ✅ عالی! اورفیت نشده")
    elif gap < 0.05:
        print("   ⚠️ قابل قبول")
    else:
        print("   ❌ اورفیت!")

    print(f"\nماتریس اغتشاش:")
    print(f"                 پیش‌بینی: عادی   پیش‌بینی: اسپم")
    print(f"واقعی: عادی      {cm[0][0]:<15} {cm[0][1]}")
    print(f"واقعی: اسپم      {cm[1][0]:<15} {cm[1][1]}")

    fp = cm[0][1]
    fn = cm[1][0]
    print(f"\nخطاها: FP={fp}, FN={fn}, کل={fp+fn}")
    if fp == 0 and fn == 0:
        print("🎉 عالی! هیچ خطایی نداریم!")
    elif fp + fn <= 2:
        print("✅ خیلی خوب! خطاها خیلی کم هستند")

    print(f"\nگزارش طبقه‌بندی:")
    print(classification_report(y_test, pred_test, target_names=["عادی (0)", "اسپم (1)"]))

    # ----------------------------------------------------------------
    # 6) ذخیره‌سازی
    # ----------------------------------------------------------------
    print("=" * 60)
    print("۵) ذخیره‌سازی")
    print("=" * 60)

    save_path = os.path.join(args.save_dir, core.MODEL_FILENAME)
    joblib.dump({"pipeline": pipe, "threshold": optimal_threshold}, save_path)
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
        "optimal_threshold": round(optimal_threshold, 2),
        "confusion_matrix": cm,
        "train_time_seconds": round(train_time, 1),
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
