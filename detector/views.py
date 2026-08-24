import os
import json
import csv
import io
import joblib
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix
)


def clean_text(text):
    """پیش‌پردازش ساده متن فارسی"""
    import re
    if not isinstance(text, str):
        return ""
    # حذف کاراکترهای اضافی
    text = re.sub(r'[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def home(request):
    """صفحه اصلی"""
    model_exists = os.path.exists(settings.MODEL_PATH)
    dataset_exists = os.path.exists(settings.DATASET_PATH)
    
    # خواندن اطلاعات مدل ذخیره شده
    model_info = None
    if model_exists:
        try:
            with open(settings.BASE_DIR / 'models' / 'model_info.json', 'r') as f:
                model_info = json.load(f)
        except:
            pass
    
    # خواندن نمونه دیتاست
    dataset_preview = []
    if dataset_exists:
        try:
            df = pd.read_csv(settings.DATASET_PATH, nrows=5)
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
    if not os.path.exists(settings.DATASET_PATH):
        return render(request, 'train.html', {'error': 'فایل دیتاست یافت نشد!'})
    
    if request.method == 'POST':
        # خواندن دیتاست
        df = pd.read_csv(settings.DATASET_PATH)
        
        # پیش‌پردازش
        df['cleaned'] = df['text'].apply(clean_text)
        df = df[df['cleaned'].str.strip() != '']
        
        # تقسیم داده
        X = df['cleaned']
        y = df['label']
        
        test_size = float(request.POST.get('test_size', 0.2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # ساخت TF-IDF
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # آموزش مدل
        model = LinearSVC(C=1.0, max_iter=10000, random_state=42, class_weight='balanced')
        model.fit(X_train_tfidf, y_train)
        
        # ارزیابی
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam')
        recall = recall_score(y_test, y_pred, pos_label='spam')
        f1 = f1_score(y_test, y_pred, pos_label='spam')
        
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # ذخیره مدل
        os.makedirs(settings.BASE_DIR / 'models', exist_ok=True)
        joblib.dump(model, settings.MODEL_PATH)
        joblib.dump(vectorizer, settings.VECTORIZER_PATH)
        
        # ذخیره اطلاعات مدل
        model_info = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_samples': len(df),
            'features': X_train_tfidf.shape[1],
            'confusion_matrix': cm.tolist(),
            'report': report,
        }
        
        with open(settings.BASE_DIR / 'models' / 'model_info.json', 'w') as f:
            json.dump(model_info, f)
        
        context = {
            'success': True,
            'model_info': model_info,
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
        
        # خواندن فایل
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
                # فایل متنی
                content = file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            # پیش‌بینی
            results = []
            spam_count = 0
            ham_count = 0
            
            for text in texts[:1000]:  # حداکثر 1000 خط
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
    """پیش‌بینی یک متن"""
    if not os.path.exists(settings.MODEL_PATH):
        return {'error': 'مدل آموزش داده نشده', 'label': None, 'confidence': 0}
    
    model = joblib.load(settings.MODEL_PATH)
    vectorizer = joblib.load(settings.VECTORIZER_PATH)
    
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    
    # محاسبه اطمینان
    if hasattr(model, 'decision_function'):
        decision = model.decision_function(features)[0]
        confidence = abs(float(decision))
        confidence = min(99, max(55, 50 + confidence * 50))
    else:
        confidence = 75.0
    
    return {
        'label': prediction,
        'label_fa': 'اسپم' if prediction == 'spam' else 'عادی',
        'confidence': round(confidence, 1),
        'cleaned_text': cleaned[:100],
    }
