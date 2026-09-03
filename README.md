# سیستم تشخیص اسپم فارسی

اپ جنگو برای تشخیص ایمیل‌های اسپم فارسی با مدل **آنسامبل نسخه ۲** (Logistic Regression + LinearSVC کالیبره + ComplementNB).

## دقت مدل (روی ۲۰۰ نمونه تست دیده‌نشده)

| معیار | مقدار |
|---|---|
| Accuracy | **۹۸.۵٪** |
| F1-Score | ۹۸.۵٪ |
| ROC-AUC | **۹۹.۹٪** |

آستانه تصمیم به‌صورت خودکار و بدون نشت به داده تست (با احتمالات out-of-fold روی داده آموزش) انتخاب می‌شود.

## نصب و اجرا

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

سپس به `http://localhost:8000` بروید.

## بخش‌های اپ

| مسیر | کاربرد |
|---|---|
| `/` | داشبورد: وضعیت سیستم، آمار مدل، ماتریس اغتشاش |
| `/train/` | آموزش مدل آنسامبل + گزارش کامل |
| `/test/` | تست تکی متن |
| `/test-file/` | آپلود فایل CSV و ارزیابی گروهی |
| `/api/predict/` | API با فرمت `POST {"text": "..."}` |

## آموزش از خط فرمان

```bash
python train_and_evaluate.py --data data/emails.csv
```

## ساختار پروژه

```
├── config/               # تنظیمات جنگو
├── detector/
│   ├── views.py          # ویوها (آموزش، تست، API)
│   └── ml/core.py        # هسته یادگیری ماشین (پیش‌پردازش، ویژگی‌ها، آنسامبل)
├── data/emails.csv       # دیتاست (۱۰۰۰ ایمیل: ۵۰۰ اسپم + ۵۰۰ عادی)
├── saved_models/         # مدل و متادیتای ذخیره‌شده
├── templates/            # قالب‌های رابط کاربری
└── train_and_evaluate.py # اسکریپت آموزش/ارزیابی خط فرمان
```
