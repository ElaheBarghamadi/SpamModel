# -*- coding: utf-8 -*-
import os
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# اضافه کردن مسیر پروژه
sys.path.insert(0, settings.BASE_DIR)

# وارد کردن کلاسیفایر
from persian_spam_classifier import (
    normalize_persian_text, load_and_clean_dataset,
    downweight_placeholder_columns, PLACEHOLDER_TOKENS, PLACEHOLDER_WEIGHT_FACTOR,
    RANDOM_STATE
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def home(request):
    """صفحه اصلی"""
    model_path = os.path.join(settings.BASE_DIR, 'saved_models', 'spam_model.joblib')
    metadata_path = os.path.join(settings.BASE_DIR, 'saved_models', 'metadata.json')
    dataset_path = os.path.join(settings.BASE_DIR, 'data', 'emails.csv')

    model_exists = os.path.exists(model_path)
    dataset_exists = os.path.exists(dataset_path)

    model_info = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        except:
            pass

    dataset_preview = []
    if dataset_exists:
        try:
            df = pd.read_csv(dataset_path, nrows=5)
            dataset_preview = df.to_dict('records')
        except:
            pass

    context = {
        'model_exists': model_exists,
        'dataset_exists': dataset_exists,
        'model_info': model_info,
        'dataset_preview': dataset_preview,
    }
    return render(request, 'home.html', context)


def train_model(request):
    """آموزش مدل با کد اصلی persian_spam_classifier"""
    dataset_path = os.path.join(settings.BASE_DIR, 'data', 'emails.csv')
    if not os.path.exists(dataset_path):
        return render(request, 'train.html', {'error': 'فایل دیتاست یافت نشد!'})

    if request.method == 'POST':
        # بارگذاری و پاک‌سازی داده
        df = load_and_clean_dataset(dataset_path)

        # کدگذاری برچسب
        positive_label_candidates = [l for l in df["label"].unique() if "spam" in l]
        positive_label = positive_label_candidates[0] if positive_label_candidates else sorted(df["label"].unique())[0]
        y = (df["label"] == positive_label).astype(int).values
        X_text = df["clean_text"].values

        # تقسیم داده
        test_size = float(request.POST.get('test_size', 0.2))
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        # استخراج ویژگی TF-IDF
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

        # کنترل وزن توکن‌های مصنوعی
        X_train_tfidf, _ = downweight_placeholder_columns(
            X_train_tfidf, feature_names, PLACEHOLDER_TOKENS.values(), PLACEHOLDER_WEIGHT_FACTOR
        )
        X_test_tfidf, _ = downweight_placeholder_columns(
            X_test_tfidf, feature_names, PLACEHOLDER_TOKENS.values(), PLACEHOLDER_WEIGHT_FACTOR
        )

        # جست‌وجوی ابرپارامتر
        param_grid = [
            {"penalty": ["l2"], "solver": ["liblinear"], "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]},
            {"penalty": ["l1"], "solver": ["liblinear"], "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]},
        ]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        base_model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)

        t0 = time.time()
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
        train_time = time.time() - t0

        # اعتبارسنجی متقاطع
        cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=cv, scoring="f1")

        # ارزیابی
        y_pred = best_model.predict(X_test_tfidf)
        y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]
        y_pred_train = best_model.predict(X_train_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        acc_train = accuracy_score(y_train, y_pred_train)
        gap = acc_train - acc

        report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0)

        # ذخیره مدل
        os.makedirs(os.path.join(settings.BASE_DIR, 'saved_models'), exist_ok=True)
        save_path = os.path.join(settings.BASE_DIR, 'saved_models', 'spam_model.joblib')
        joblib.dump({
            "vectorizer": vectorizer,
            "model": best_model,
            "feature_names": feature_names,
        }, save_path)

        # ذخیره متادیتا
        metadata = {
            "display_name": "Logistic Regression (GridSearchCV)",
            "n_samples_total": len(df),
            "n_train": len(X_train_text),
            "n_test": len(X_test_text),
            "test_accuracy": round(acc, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1": round(f1, 4),
            "test_roc_auc": round(roc_auc, 4),
            "train_accuracy": round(acc_train, 4),
            "overfit_gap": round(gap, 4),
            "best_params": str(grid.best_params_),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "confusion_matrix": cm,
            "train_time_seconds": round(train_time, 1),
        }
        metadata_path = os.path.join(settings.BASE_DIR, 'saved_models', 'metadata.json')
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # تبدیل کلید f1-score
        for key in report:
            if isinstance(report[key], dict) and 'f1-score' in report[key]:
                report[key]['f1_score'] = report[key].pop('f1-score')

        context = {
            'success': True,
            'model_info': metadata,
            'report': report,
        }
        return render(request, 'train.html', context)

    return render(request, 'train.html')


def test_text(request):
    """تست با متن"""
    result = None
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text:
            result = predict_text(text)
            result['input_text'] = text
    return render(request, 'test_text.html', {'result': result})


def test_file(request):
    """تست با فایل"""
    results = None
    summary = None
    metrics = None
    has_labels = False

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                df = pd.read_excel(file)
            else:
                content = file.read().decode('utf-8')
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                df = pd.DataFrame({'text': lines})

            text_col = None
            for col in ['text', 'message', 'content', 'comment', 'متن', 'پیام']:
                if col in df.columns:
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[0]

            label_col = None
            for col in ['label', 'labels', 'class', 'target', 'برچسب', 'دسته']:
                if col in df.columns:
                    label_col = col
                    break

            label_map = {
                "ham": 0, "normal": 0, "not spam": 0, "0": 0, 0: 0,
                "spam": 1, "1": 1, 1: 1, "اسپم": 1, "عادی": 0,
            }

            if label_col:
                has_labels = True
                df['true_label'] = df[label_col].map(label_map)
                df = df.dropna(subset=['text', 'true_label'])
                df['true_label'] = df['true_label'].astype(int)
            else:
                df = df.dropna(subset=[text_col])

            df = df.head(2000).reset_index(drop=True)

            results = []
            y_true = []
            y_pred = []
            y_proba = []

            for idx, row in df.iterrows():
                text = str(row[text_col])
                if not text or text == 'nan':
                    continue

                pred = predict_text(text)
                pred['index'] = idx + 1
                pred['original_text'] = text[:200]
                results.append(pred)

                if has_labels:
                    true_val = int(row['true_label'])
                    pred_val = 1 if pred['label'] == 'spam' else 0
                    y_true.append(true_val)
                    y_pred.append(pred_val)
                    y_proba.append(pred.get('probability', 50) / 100)
                    pred['true_label'] = true_val
                    pred['true_label_fa'] = 'اسپم' if true_val == 1 else 'عادی'
                    pred['correct'] = true_val == pred_val

            spam_count = sum(1 for r in results if r['label'] == 'spam')
            ham_count = len(results) - spam_count

            summary = {
                'total': len(results),
                'spam': spam_count,
                'ham': ham_count,
                'spam_percent': round(spam_count / len(results) * 100, 1) if results else 0,
                'ham_percent': round(ham_count / len(results) * 100, 1) if results else 0,
                'has_labels': has_labels,
            }

            if has_labels and len(y_true) > 0:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_true, y_proba)
                except:
                    roc_auc = 0
                cm = confusion_matrix(y_true, y_pred).tolist()
                report = classification_report(y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0)

                for key in report:
                    if isinstance(report[key], dict) and 'f1-score' in report[key]:
                        report[key]['f1_score'] = report[key].pop('f1-score')

                metrics = {
                    'accuracy': round(acc * 100, 2),
                    'precision': round(prec * 100, 2),
                    'recall': round(rec * 100, 2),
                    'f1': round(f1 * 100, 2),
                    'roc_auc': round(roc_auc * 100, 2),
                    'confusion_matrix': cm,
                    'report': report,
                }

        except Exception as e:
            return render(request, 'test_file.html', {'error': str(e)})

    context = {
        'results': results,
        'summary': summary,
        'metrics': metrics,
        'has_labels': has_labels,
    }
    return render(request, 'test_file.html', context)


@csrf_exempt
def api_predict(request):
    """API برای پیش‌بینی"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            if text:
                result = predict_text(text)
                return JsonResponse(result)
            return JsonResponse({'error': 'متن ارسال نشده'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON نامعتبر'}, status=400)
    return JsonResponse({'error': 'فقط POST'}, status=405)


def predict_text(text):
    """پیش‌بینی با مدل آموزش‌دیده"""
    model_path = os.path.join(settings.BASE_DIR, 'saved_models', 'spam_model.joblib')

    if not os.path.exists(model_path):
        return {'error': 'مدل آموزش داده نشده', 'label': None, 'confidence': 0}

    saved = joblib.load(model_path)
    vectorizer = saved["vectorizer"]
    model = saved["model"]
    feature_names = saved["feature_names"]

    # نرمال‌سازی متن
    clean = normalize_persian_text(text)

    # استخراج ویژگی
    X_tfidf = vectorizer.transform([clean])

    # کنترل وزن توکن‌های مصنوعی
    X_tfidf, _ = downweight_placeholder_columns(
        X_tfidf, feature_names, PLACEHOLDER_TOKENS.values(), PLACEHOLDER_WEIGHT_FACTOR
    )

    # پیش‌بینی
    proba = model.predict_proba(X_tfidf)[:, 1][0]
    label = "spam" if proba >= 0.5 else "ham"
    confidence = round(abs(proba - 0.5) * 200, 1)
    confidence = min(99, max(50, confidence))

    return {
        'label': label,
        'label_fa': 'اسپم' if label == 'spam' else 'عادی',
        'confidence': confidence,
        'probability': round(proba * 100, 1),
    }
