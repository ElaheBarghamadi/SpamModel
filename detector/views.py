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

        # تبدیل label
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

        # آموزش
        t0 = time.time()
        clf = core.get_classifier()
        pipe = core.build_pipeline(clf)
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # ارزیابی
        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_test)
        prec = precision_score(y_test, pred_test)
        rec = recall_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test)
        roc_auc = roc_auc_score(y_test, proba_test)
        cm = confusion_matrix(y_test, pred_test).tolist()

        report = classification_report(y_test, pred_test, target_names=["ham", "spam"], output_dict=True)

        # ذخیره مدل
        os.makedirs(os.path.join(settings.BASE_DIR, 'saved_models'), exist_ok=True)
        save_path = os.path.join(settings.BASE_DIR, 'saved_models', core.MODEL_FILENAME)
        joblib.dump({"pipeline": pipe, "threshold": 0.5}, save_path)

        # ذخیره متادیتا
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

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']

        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                if 'text' in df.columns:
                    texts = df['text'].tolist()
                elif len(df.columns) >= 1:
                    texts = df.iloc[:, 0].tolist()
                else:
                    return render(request, 'test_file.html', {'error': 'ستون text یافت نشد'})
            else:
                content = file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]

            results = []
            spam_count = 0
            ham_count = 0

            for text in texts[:1000]:
                if text and str(text) != 'nan':
                    pred = predict_text(str(text))
                    results.append(pred)
                    if pred['label'] == 'spam':
                        spam_count += 1
                    else:
                        ham_count += 1

            summary = {
                'total': len(results),
                'spam': spam_count,
                'ham': ham_count,
                'spam_percent': round(spam_count / len(results) * 100, 1) if results else 0,
                'ham_percent': round(ham_count / len(results) * 100, 1) if results else 0,
            }
        except Exception as e:
            return render(request, 'test_file.html', {'error': str(e)})

    return render(request, 'test_file.html', {'results': results, 'summary': summary})


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
    confidence = round(abs(proba - 0.5) * 200, 1)  # تبدیل به درصد اطمینان
    confidence = min(99, max(50, confidence))

    return {
        'label': label,
        'label_fa': 'اسپم' if label == 'spam' else 'عادی',
        'confidence': confidence,
        'probability': round(proba * 100, 1),
    }
