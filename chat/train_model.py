"""
آموزش مدل تشخیص اسپم فارسی با دیتاست جدید
"""

import os
import sys
import django
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# تنظیم Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from chat.preprocess import clean_text


def train():
    print("📊 بارگذاری دیتاست...")
    
    # خواندن دیتاست
    df = pd.read_csv('data_new/emails.csv')
    print(f"   تعداد کل: {len(df)}")
    print(f"   توزیع: {df['label'].value_counts().to_dict()}")
    
    # پیش‌پردازش
    print("\n🔧 پیش‌پردازش متن‌ها...")
    df['cleaned'] = df['text'].apply(clean_text)
    
    # حذف ردیف‌های خالی
    df = df[df['cleaned'].str.strip() != '']
    print(f"   تعداد بعد از پاکسازی: {len(df)}")
    
    # تقسیم داده
    X = df['cleaned']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 تقسیم داده:")
    print(f"   آموزش: {len(X_train)}")
    print(f"   تست: {len(X_test)}")
    
    # ساخت TF-IDF
    print("\n🔤 ساخت ویژگی‌های TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   تعداد ویژگی‌ها: {X_train_tfidf.shape[1]}")
    
    # آموزش مدل
    print("\n🤖 آموزش مدل LinearSVC...")
    model = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42,
        class_weight='balanced',
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # ارزیابی
    print("\n📊 ارزیابی مدل:")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   دقت: {accuracy:.4f}")
    print("\n   گزارش طبقه‌بندی:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # ذخیره مدل
    print("\n💾 ذخیره مدل...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spam_model_repo', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'linear_svc_model.joblib')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"   ✅ مدل ذخیره شد: {model_path}")
    print(f"   ✅ وکتورایزر ذخیره شد: {vectorizer_path}")
    
    # تست نمونه
    print("\n🧪 تست نمونه:")
    test_samples = [
        "سلام دوستان حالتون چطوره؟",
        "کانال تلگرام لینک بیو فالو کنید",
        "درآمد میلیونی بدون سرمایه! کلیک کنید",
        "خفه شو برو گمشو",
        "ممنون از راهنماییتون",
        "کد تخفیف ویژه 50 درصد فقط امروز!",
    ]
    
    for text in test_samples:
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]
        icon = "✅" if pred == "ham" else "⚠️"
        print(f"   {icon} [{pred:4s}] {text[:50]}")
    
    print("\n🎉 آموزش با موفقیت انجام شد!")
    return model, vectorizer


if __name__ == '__main__':
    train()
