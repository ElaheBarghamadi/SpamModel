# -*- coding: utf-8 -*-
import os
import json
import joblib
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.ml import core


def home(request):
    """صفحه اصلی"""
    model_path = os.path.join(settings.BASE_DIR, 'saved_models', core.MODEL_FILENAME)
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
    """آموزش مدل"""
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report
    )

    dataset_path = os.path.join(settings.BASE_DIR, 'data', 'emails.csv')
    if not os.path.exists(dataset_path):
        return render(request, 'train.html', {'error': 'فایل دیتاست یافت نشد!'})

    if request.method == 'POST':
        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

        label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1, 0: 0, 1: 1}
        df["label"] = df["label"].map(label_map)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        test_size = float(request.POST.get('test_size', 0.2))

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df["label"]
        )

        X_train = core.as_model_input(train_df["text"])
        y_train = train_df["label"].values
        X_test = core.as_model_input(test_df["text"])
        y_test = test_df["label"].values

        t0 = time.time()
        clf = core.get_classifier()
        pipe = core.build_pipeline(clf)
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_test)
        prec = precision_score(y_test, pred_test)
        rec = recall_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test)
        roc_auc = roc_auc_score(y_test, proba_test)
        cm = confusion_matrix(y_test, pred_test).tolist()

        report = classification_report(y_test, pred_test, target_names=["ham", "spam"], output_dict=True)

        os.makedirs(os.path.join(settings.BASE_DIR, 'saved_models'), exist_ok=True)
        save_path = os.path.join(settings.BASE_DIR, 'saved_models', core.MODEL_FILENAME)
        joblib.dump({"pipeline": pipe, "threshold": 0.5}, save_path)

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
        metadata_path = os.path.join(settings.BASE_DIR, 'saved_models', 'metadata.json')
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

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
    """تست با فایل - با پشتیبانی از برچسب و نمایش نمرات"""
    results = None
    summary = None
    metrics = None
    has_labels = False

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']

        try:
            # خواندن فایل
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                df = pd.read_excel(file)
            else:
                content = file.read().decode('utf-8')
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                df = pd.DataFrame({'text': lines})

            # تشخیص ستون متن
            text_col = None
            for col in ['text', 'message', 'content', 'comment', 'متن', 'پیام']:
                if col in df.columns:
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[0]

            # تشخیص ستون برچسب
            label_col = None
            for col in ['label', 'labels', 'class', 'target', 'برچسب', 'دسته']:
                if col in df.columns:
                    label_col = col
                    break

            # تبدیل برچسب‌ها
            label_map = {
                "ham": 0, "normal": 0, "not spam": 0, "0": 0, 0: 0,
                "spam": 1, "1": 1, 1: 1, "اسپم": 1, "عادی": 0,
            }

            if label_col:
                has_labels = True
                df['true_label_raw'] = df[label_col].copy()
                df['true_label'] = df[label_col].map(label_map)
                # ردیف‌های بدون برچسب معتبر رو حذف کن
                df = df.dropna(subset=['text', 'true_label'])
                df['true_label'] = df['true_label'].astype(int)
            else:
                df = df.dropna(subset=[text_col])

            # محدود کردن به 2000 ردیف
            df = df.head(2000).reset_index(drop=True)

            # پیش‌بینی
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

            # خلاصه
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

            # محاسبه نمرات اگر برچسب داشت
            if has_labels and len(y_true) > 0:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, roc_auc_score, confusion_matrix, classification_report
                )

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
                    'correct': sum(1 for c in y_true if c == y_pred[y_true.index(c)]),
                    'wrong': sum(1 for c in y_true if c != y_pred[y_true.index(c)]),
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
    """پیش‌بینی یک متن با مدل آموزش‌دیده"""
    model_path = os.path.join(settings.BASE_DIR, 'saved_models', core.MODEL_FILENAME)

    if not os.path.exists(model_path):
        return {'error': 'مدل آموزش داده نشده', 'label': None, 'confidence': 0}

    saved = joblib.load(model_path)
    pipe = saved["pipeline"]
    threshold = saved.get("threshold", 0.5)

    X = core.as_model_input([text])
    proba = pipe.predict_proba(X)[:, 1][0]
    label = "spam" if proba >= threshold else "ham"
    confidence = round(abs(proba - 0.5) * 200, 1)
    confidence = min(99, max(50, confidence))

    return {
        'label': label,
        'label_fa': 'اسپم' if label == 'spam' else 'عادی',
        'confidence': confidence,
        'probability': round(proba * 100, 1),
    }
