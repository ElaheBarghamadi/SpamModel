# -*- coding: utf-8 -*-
"""
ویوهای اپ تشخیص اسپم فارسی — نسخه تمیز
همه‌چیز از هسته مشترک detector.ml.core استفاده می‌کند
"""
import os
import json
import time
import threading

import joblib
import numpy as np
import pandas as pd

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from detector.ml import core

SAVED_DIR = os.path.join(settings.BASE_DIR, 'saved_models')
MODEL_PATH = os.path.join(SAVED_DIR, core.MODEL_FILENAME)
METADATA_PATH = os.path.join(SAVED_DIR, 'metadata.json')
DATASET_PATH = os.path.join(settings.BASE_DIR, 'data', 'emails.csv')


# ----------------------------------------------------------------
# ابزارهای مشترک
# ----------------------------------------------------------------
def load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_dataset():
    """بارگذاری و ادغام همه دیتاست‌های پوشه data"""
    return core.load_datasets(os.path.join(settings.BASE_DIR, 'data'))


def save_model(pipe, threshold, metrics):
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump({'pipeline': pipe, 'threshold': float(threshold), 'version': 'v3'}, MODEL_PATH)
    metrics['display_name'] = core.MODEL_NAME
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def predict_text(text):
    """پیش‌بینی برای یک متن با مدل آموزش‌دیده"""
    if not os.path.exists(MODEL_PATH):
        return {'error': 'مدل هنوز آموزش داده نشده است', 'label': None, 'confidence': 0}

    saved = joblib.load(MODEL_PATH)
    proba = saved['pipeline'].predict_proba(core.as_model_input([text]))[0, 1]
    threshold = float(saved.get('threshold', 0.5))

    label = 'spam' if proba >= threshold else 'ham'
    return {
        'label': label,
        'label_fa': 'اسپم' if label == 'spam' else 'عادی',
        'confidence': round(min(99, max(50, abs(proba - 0.5) * 200)), 1),
        'probability': round(float(proba) * 100, 1),
    }


# ----------------------------------------------------------------
# صفحه اصلی
# ----------------------------------------------------------------
def home(request):
    meta = load_metadata()
    dataset_exists = os.path.exists(DATASET_PATH)

    dataset_stats = None
    preview = []
    if dataset_exists:
        try:
            df = load_dataset()
            n_spam = int(df['label'].sum())
            dataset_stats = {
                'total': len(df),
                'spam': n_spam,
                'ham': len(df) - n_spam,
            }
            preview = [
                {'text': r.text[:120], 'label': 'spam' if r.label == 1 else 'ham'}
                for r in df.sample(5, random_state=1).itertuples()
            ]
        except Exception:
            pass

    context = {
        'model_exists': os.path.exists(MODEL_PATH),
        'dataset_exists': dataset_exists,
        'dataset_stats': dataset_stats,
        'preview': preview,
    }
    if meta:
        context.update({
            'model_name': meta.get('display_name', core.MODEL_NAME),
            'stats': {
                'accuracy': round(meta.get('test_accuracy', 0) * 100, 2),
                'precision': round(meta.get('test_precision', 0) * 100, 2),
                'recall': round(meta.get('test_recall', 0) * 100, 2),
                'f1': round(meta.get('test_f1', 0) * 100, 2),
                'roc_auc': round(meta.get('test_roc_auc', 0) * 100, 2),
            },
            'meta': meta,
        })
    return render(request, 'home.html', context)


# ----------------------------------------------------------------
# آموزش مدل — با لاگ زنده
# ----------------------------------------------------------------
TRAIN_STAGES = [
    'بارگذاری و آماده‌سازی داده‌ها',
    'تقسیم داده به آموزش و تست',
    'آموزش مدل (LinearSVC کالیبره)',
    'بهینه‌سازی آستانه (OOF)',
    'ارزیابی روی داده تست',
    'ذخیره مدل و متادیتا',
]

TRAIN_LOCK = threading.Lock()
TRAIN_STATE = {
    'status': 'idle',      # idle | running | done | error
    'stage': 0,
    'logs': [],
    'started_at': None,
    'result': None,
    'error': None,
}


def _tlog(msg):
    stamp = time.strftime('%H:%M:%S')
    with TRAIN_LOCK:
        TRAIN_STATE['logs'].append({'t': stamp, 'm': msg})


def _set_stage(i):
    with TRAIN_LOCK:
        TRAIN_STATE['stage'] = i
    _tlog(f'── مرحله {i + 1} از {len(TRAIN_STAGES)}: {TRAIN_STAGES[i]}')


def train_worker(test_size):
    """آموزش در نخ پس‌زمینه با لاگ‌گیری مرحله‌به‌مرحله"""
    try:
        _set_stage(0)
        df = load_dataset()
        n_spam = int(df['label'].sum())
        _tlog(f'✓ دیتاست بارگذاری شد: {len(df)} نمونه ({len(df) - n_spam} عادی، {n_spam} اسپم)')
        df = df.dropna(subset=['text']).reset_index(drop=True)

        _set_stage(1)
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=core.RANDOM_STATE, stratify=df['label']
        )
        X_train = core.as_model_input(train_df['text'])
        y_train = train_df['label'].values
        X_test = core.as_model_input(test_df['text'])
        y_test = test_df['label'].values
        _tlog(f'✓ تقسیم انجام شد: آموزش {len(train_df)} | تست {len(test_df)} (سهم تست: {test_size})')

        _set_stage(2)
        _tlog('→ مدل: LinearSVC کالیبره (C=1.0) با ویژگی‌های TF-IDF کلمه‌ای و کاراکتری')
        t0 = time.time()
        pipe = core.build_pipeline(core.get_classifier())
        pipe.fit(X_train, y_train)
        _tlog(f'✓ آموزش مدل تمام شد ({time.time() - t0:.1f} ثانیه)')

        _set_stage(3)
        _tlog('→ اجرای اعتبارسنجی متقاطع ۵ لایه روی داده آموزش (بدون نشت به تست)...')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=core.RANDOM_STATE)
        t0 = time.time()
        proba_oof = cross_val_predict(pipe, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
        threshold = core.find_optimal_threshold_from_proba(proba_oof, y_train)
        _tlog(f'✓ آستانه بهینه پیدا شد: {threshold:.2f} ({time.time() - t0:.1f} ثانیه)')

        _set_stage(4)
        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= threshold).astype(int)
        pred_train = (pipe.predict_proba(X_train)[:, 1] >= threshold).astype(int)

        acc = accuracy_score(y_test, pred_test)
        acc_train = accuracy_score(y_train, pred_train)
        m = {
            'n_samples_total': len(df),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'test_accuracy': round(acc, 4),
            'test_precision': round(precision_score(y_test, pred_test), 4),
            'test_recall': round(recall_score(y_test, pred_test), 4),
            'test_f1': round(f1_score(y_test, pred_test), 4),
            'test_roc_auc': round(roc_auc_score(y_test, proba_test), 4),
            'train_accuracy': round(acc_train, 4),
            'overfit_gap': round(acc_train - acc, 4),
            'optimal_threshold': round(float(threshold), 2),
            'confusion_matrix': confusion_matrix(y_test, pred_test).tolist(),
            'train_time_seconds': 0,  # در انتهای کار با زمان کل پر می‌شود
        }
        _tlog(f'✓ Accuracy={m["test_accuracy"]:.4f} | F1={m["test_f1"]:.4f} | AUC={m["test_roc_auc"]:.4f}')
        cm = m['confusion_matrix']
        _tlog(f'✓ ماتریس اغتشاش: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}')

        _set_stage(5)
        m['train_time_seconds'] = round(time.time() - TRAIN_STATE['started_at'], 1)
        save_model(pipe, threshold, m)
        _tlog('✓ مدل در saved_models/ensemble_model.joblib ذخیره شد')
        _tlog('🎉 آموزش با موفقیت کامل شد!')

        report = {}
        raw_report = classification_report(
            y_test, pred_test, target_names=['عادی', 'اسپم'], output_dict=True, zero_division=0
        )
        for key, name in [('عادی', 'عادی (0)'), ('اسپم', 'اسپم (1)')]:
            r = raw_report.get(key, {})
            report[name] = {
                'precision': round(r.get('precision', 0) * 100, 2),
                'recall': round(r.get('recall', 0) * 100, 2),
                'f1': round(r.get('f1-score', 0) * 100, 2),
                'support': r.get('support', 0),
            }

        context = {
            'success': True,
            'metrics': {
                'accuracy': round(m['test_accuracy'] * 100, 2),
                'precision': round(m['test_precision'] * 100, 2),
                'recall': round(m['test_recall'] * 100, 2),
                'f1': round(m['test_f1'] * 100, 2),
                'roc_auc': round(m['test_roc_auc'] * 100, 2),
            },
            'raw': m,
            'cm': m['confusion_matrix'],
            'report': report,
            'threshold': m['optimal_threshold'],
            'train_time': m['train_time_seconds'],
            'overfit_gap': round(m['overfit_gap'] * 100, 2),
        }
        with TRAIN_LOCK:
            TRAIN_STATE['result'] = context
            TRAIN_STATE['status'] = 'done'
    except Exception as e:
        _tlog(f'✗ خطا: {e}')
        with TRAIN_LOCK:
            TRAIN_STATE['status'] = 'error'
            TRAIN_STATE['error'] = str(e)


@csrf_exempt
def train_start(request):
    """شروع آموزش در پس‌زمینه"""
    if request.method != 'POST':
        return JsonResponse({'error': 'فقط متد POST'}, status=405)
    with TRAIN_LOCK:
        if TRAIN_STATE['status'] == 'running':
            return JsonResponse({'started': False, 'already_running': True})
        TRAIN_STATE.update({
            'status': 'running', 'stage': 0, 'logs': [],
            'result': None, 'error': None, 'started_at': time.time(),
        })

    try:
        test_size = min(0.5, max(0.1, float(request.POST.get('test_size', 0.2))))
    except (ValueError, TypeError):
        test_size = 0.2

    threading.Thread(target=train_worker, args=(test_size,), daemon=True).start()
    return JsonResponse({'started': True, 'stages': TRAIN_STAGES})


@csrf_exempt
def train_status(request):
    """وضعیت و لاگ‌های آموزش برای نظرسنجی مرورگر"""
    with TRAIN_LOCK:
        started = TRAIN_STATE['started_at']
        return JsonResponse({
            'status': TRAIN_STATE['status'],
            'stage': TRAIN_STATE['stage'],
            'stages': TRAIN_STAGES,
            'logs': TRAIN_STATE['logs'],
            'elapsed': round(time.time() - started, 1) if started else 0,
            'error': TRAIN_STATE['error'],
        })


def train_model(request):
    if not os.path.isdir(os.path.join(settings.BASE_DIR, 'data')):
        return render(request, 'train.html', {'error': 'پوشه دیتاست (data/) یافت نشد!',
                                              'stages': TRAIN_STAGES})

    context = {'stages': TRAIN_STAGES}
    with TRAIN_LOCK:
        context['live_status'] = TRAIN_STATE['status']
        if TRAIN_STATE['status'] == 'done' and TRAIN_STATE['result']:
            context.update(TRAIN_STATE['result'])
        elif TRAIN_STATE['status'] == 'error' and TRAIN_STATE['error']:
            context['train_error'] = TRAIN_STATE['error']
    return render(request, 'train.html', context)


# ----------------------------------------------------------------
# تست متن
# ----------------------------------------------------------------
def test_text(request):
    result = None
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if text:
            result = predict_text(text)
            result['input_text'] = text
    return render(request, 'test_text.html', {'result': result})


# ----------------------------------------------------------------
# تست فایل
# ----------------------------------------------------------------
LABEL_MAP = {
    'ham': 0, 'normal': 0, 'not spam': 0, '0': 0, 0: 0, 'عادی': 0,
    'spam': 1, '1': 1, 1: 1, 'اسپم': 1,
}


def test_file(request):
    results = summary = metrics = None
    has_labels = False

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            if file.name.lower().endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                lines = [l.strip() for l in file.read().decode('utf-8').split('\n') if l.strip()]
                df = pd.DataFrame({'text': lines})

            text_col = next((c for c in ['text', 'message', 'content', 'comment', 'متن', 'پیام']
                             if c in df.columns), df.columns[0])
            label_col = next((c for c in ['label', 'labels', 'class', 'target', 'برچسب', 'دسته']
                              if c in df.columns), None)

            if label_col:
                has_labels = True
                df['true_label'] = df[label_col].map(LABEL_MAP)
                df = df.dropna(subset=[text_col, 'true_label'])
                df['true_label'] = df['true_label'].astype(int)
            else:
                df = df.dropna(subset=[text_col])

            df = df.head(2000).reset_index(drop=True)

            texts = [str(t) for t in df[text_col] if str(t) and str(t) != 'nan']
            saved = joblib.load(MODEL_PATH)
            probas = saved['pipeline'].predict_proba(core.as_model_input(texts))[:, 1]
            threshold = float(saved.get('threshold', 0.5))

            results = []
            y_true, y_pred, y_proba = [], [], []
            for i, (idx, row) in enumerate(df.iterrows()):
                text = str(row[text_col])
                if not text or text == 'nan':
                    continue
                proba = float(probas[len(results)])
                label = 'spam' if proba >= threshold else 'ham'
                item = {
                    'index': len(results) + 1,
                    'original_text': text[:200],
                    'label': label,
                    'label_fa': 'اسپم' if label == 'spam' else 'عادی',
                    'probability': round(proba * 100, 1),
                }
                if has_labels:
                    true_val = int(row['true_label'])
                    item['true_label_fa'] = 'اسپم' if true_val == 1 else 'عادی'
                    item['correct'] = true_val == (1 if label == 'spam' else 0)
                    y_true.append(true_val)
                    y_pred.append(1 if label == 'spam' else 0)
                    y_proba.append(proba)
                results.append(item)

            spam_count = sum(1 for r in results if r['label'] == 'spam')
            summary = {
                'total': len(results),
                'spam': spam_count,
                'ham': len(results) - spam_count,
                'spam_percent': round(spam_count / max(len(results), 1) * 100, 1),
                'ham_percent': round((len(results) - spam_count) / max(len(results), 1) * 100, 1),
                'has_labels': has_labels,
            }

            if has_labels and y_true:
                cm = confusion_matrix(y_true, y_pred).tolist()
                metrics = {
                    'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
                    'precision': round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
                    'recall': round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
                    'f1': round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
                    'roc_auc': round(roc_auc_score(y_true, y_proba) * 100, 2) if len(set(y_true)) > 1 else None,
                    'confusion_matrix': cm,
                }

        except Exception as e:
            return render(request, 'test_file.html', {'error': f'خطا در پردازش فایل: {e}'})

    return render(request, 'test_file.html', {
        'results': results, 'summary': summary, 'metrics': metrics, 'has_labels': has_labels,
    })


# ----------------------------------------------------------------
# API
# ----------------------------------------------------------------
@csrf_exempt
def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'فقط متد POST پشتیبانی می‌شود'}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'JSON نامعتبر'}, status=400)

    text = data.get('text', '')
    if not text:
        return JsonResponse({'error': 'فیلد text ارسال نشده'}, status=400)
    return JsonResponse(predict_text(text))
