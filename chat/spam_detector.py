"""
سرویس تشخیص اسپم و محتوای نامناسب
بر اساس مدل PHICAD - Persian Harmful Comment Classifier
"""

import re
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# الگوهای پیش‌پردازش متن فارسی
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
USERNAME_PATTERN = re.compile(r"@\w+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "\U0001F000-\U0001F0FF"
    "]+",
    flags=re.UNICODE,
)
EXTRA_SPACE_PATTERN = re.compile(r"\s+")

PERSIAN_CHAR_MAP = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "ٲ": "ا",
    "ﻻ": "لا", "ﷲ": "الله",
    "‌": " ", "‏": "", "‎": "",
}

PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
LETTER_ELONGATION_PATTERN = re.compile(r"([\u0600-\u06FF])\1{2,}")

_PERSIAN_LETTER = r"[\u0600-\u06FF]"
SEPARATOR_OBFUSCATION_PATTERN = re.compile(
    rf"(?<!{_PERSIAN_LETTER})(?:{_PERSIAN_LETTER}[.*_\-]+){{1,}}{_PERSIAN_LETTER}(?!{_PERSIAN_LETTER})"
)
SPACED_OBFUSCATION_PATTERN = re.compile(
    rf"(?<!{_PERSIAN_LETTER})(?:{_PERSIAN_LETTER}\s+){{2,}}{_PERSIAN_LETTER}(?!{_PERSIAN_LETTER})"
)
FULLY_CENSORED_TOKEN_PATTERN = re.compile(r"(?<!\S)[*#@$%_\-]{2,}(?!\S)")
CENSORED_TOKEN_PLACEHOLDER = "سانسور"

ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF0-9a-zA-Z\s.!؟?]"
)


def clean_text(text: str) -> str:
    """پیش‌پردازش کامل متن فارسی"""
    if not isinstance(text, str):
        return ""

    text = URL_PATTERN.sub(" ", text)
    text = USERNAME_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)

    for arabic_char, persian_char in PERSIAN_CHAR_MAP.items():
        text = text.replace(arabic_char, persian_char)

    text = text.translate(PERSIAN_DIGIT_MAP)
    text = text.translate(ARABIC_DIGIT_MAP)
    text = DIACRITICS_PATTERN.sub("", text)
    text = LETTER_ELONGATION_PATTERN.sub(r"\1", text)

    def _strip_separators(match):
        return re.sub(r"[\s.*_\-]+", "", match.group(0))

    text = SEPARATOR_OBFUSCATION_PATTERN.sub(_strip_separators, text)
    text = SPACED_OBFUSCATION_PATTERN.sub(_strip_separators, text)
    text = FULLY_CENSORED_TOKEN_PATTERN.sub(CENSORED_TOKEN_PLACEHOLDER, text)
    text = ALLOWED_CHARS_PATTERN.sub(" ", text)
    text = EXTRA_SPACE_PATTERN.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# کلمات کلیدی اسپم و نفرت‌انگیز فارسی
# ---------------------------------------------------------------------------

SPAM_KEYWORDS = [
    'کانال تلگرام', 'لینک بیو', 'فالو کنید', 'فالو بک', 'سابسکرایب',
    'لایک کنید', 'کد تخفیف', 'خرید کنید', 'فروش ویژه', 'تخفیف ویژه',
    'درآمد میلیونی', 'کسب درآمد', 'پولدار شوید', 'بدون سرمایه',
    'ارسال رایگان', 'گارانتی', 'ضمانت', 'محدود', 'فوری',
    'واتساپ', 'تلگرام', 'اینستاگرام', 'کلیک کنید', 'ثبت نام کنید',
    'جایزه ببرید', 'برنده شدید', 'هدیه رایگان', 'شانس شما',
    '091', '093', '090', '092', '099',
    'تومان', 'ریال', 'قیمت', 'ارزان', 'گران',
]

HATE_KEYWORDS = [
    'احمق', 'بی‌عقل', 'نادان', 'کودن', 'اسکل', 'بی‌شرف',
    'لعنت', 'بی‌ناموس', 'پدرسوخته', 'لاشی', 'خایه',
    'بی‌غیرت', 'جوجه', 'بوزینه', 'خر', 'الاغ',
    'خفه شو', 'برو گمشو', 'خفه', 'گمشو', 'برو بابا',
    'بی‌خود', 'مزخرف', 'چرت', 'پرت', 'مسخره',
    'کثافت', 'کثیف', 'چندش', 'حال‌بهم‌زن',
]


class SpamDetector:
    """تشخیص اسپم و محتوای نامناسب در پیام‌های فارسی"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """بارگذاری مدل آموزش‌دیده"""
        try:
            import joblib
            from django.conf import settings

            model_path = os.path.join(settings.BASE_DIR, 'spam_model_repo', 'models', 'linear_svc_model.joblib')
            vectorizer_path = os.path.join(settings.BASE_DIR, 'spam_model_repo', 'models', 'tfidf_vectorizer.joblib')

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.model_loaded = True
                logger.info("✅ مدل اسپم با موفقیت بارگذاری شد")
            else:
                logger.warning("⚠️ فایل‌های مدل یافت نشد. از سیستم کلمات کلیدی استفاده می‌شود")

        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل: {e}")

    def analyze_text(self, text: str) -> dict:
        """
        تحلیل متن و تشخیص اسپم/نفرت

        خروجی:
        {
            'is_harmful': bool,
            'label': 'Normal' | 'Spam' | 'Hate',
            'confidence': float,
            'warning_message': str,
            'warning_type': 'spam' | 'hate' | 'normal',
        }
        """
        if not text or not text.strip():
            return self._normal_result()

        cleaned = clean_text(text)

        # اگر مدل آموزش‌دیده موجود باشد
        if self.model_loaded:
            return self._predict_with_model(cleaned, text)

        # در غیر این صورت از سیستم کلمات کلیدی استفاده کن
        return self._predict_with_keywords(cleaned, text)

    def _predict_with_model(self, cleaned_text: str, original_text: str) -> dict:
        """پیش‌بینی با مدل آموزش‌دیده"""
        try:
            features = self.vectorizer.transform([cleaned_text])
            prediction = self.model.predict(features)[0]

            # دریافت امتیاز اطمینان
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                confidence = float(max(decision))
            else:
                confidence = 0.8

            if prediction == 'Hate':
                return {
                    'is_harmful': True,
                    'label': 'Hate',
                    'confidence': min(confidence, 1.0),
                    'warning_message': '⚠️ این پیام شامل محتوای نفرت‌انگیز یا توهین‌آمیز است.',
                    'warning_type': 'hate',
                    'original_text': original_text,
                }
            elif prediction == 'Spam':
                return {
                    'is_harmful': True,
                    'label': 'Spam',
                    'confidence': min(confidence, 1.0),
                    'warning_message': '⚠️ این پیام به عنوان اسپم شناسایی شد.',
                    'warning_type': 'spam',
                    'original_text': original_text,
                }
            else:
                return self._normal_result()

        except Exception as e:
            logger.error(f"خطا در پیش‌بینی مدل: {e}")
            return self._predict_with_keywords(cleaned_text, original_text)

    def _predict_with_keywords(self, cleaned_text: str, original_text: str) -> dict:
        """پیش‌بینی با سیستم کلمات کلیدی (جایگزین)"""
        text_lower = cleaned_text.lower()

        # بررسی نفرت
        hate_score = 0
        hate_matches = []
        for keyword in HATE_KEYWORDS:
            if keyword in text_lower:
                hate_score += 1
                hate_matches.append(keyword)

        # بررسی اسپم
        spam_score = 0
        spam_matches = []
        for keyword in SPAM_KEYWORDS:
            if keyword in text_lower:
                spam_score += 1
                spam_matches.append(keyword)

        # اولویت با نفرت است
        if hate_score >= 1:
            return {
                'is_harmful': True,
                'label': 'Hate',
                'confidence': min(0.5 + (hate_score * 0.15), 0.95),
                'warning_message': '⚠️ این پیام شامل محتوای نفرت‌انگیز یا توهین‌آمیز است.',
                'warning_type': 'hate',
                'original_text': original_text,
                'matched_keywords': hate_matches,
            }

        if spam_score >= 2:
            return {
                'is_harmful': True,
                'label': 'Spam',
                'confidence': min(0.5 + (spam_score * 0.1), 0.9),
                'warning_message': '⚠️ این پیام به عنوان اسپم شناسایی شد.',
                'warning_type': 'spam',
                'original_text': original_text,
                'matched_keywords': spam_matches,
            }

        return self._normal_result()

    def _normal_result(self) -> dict:
        return {
            'is_harmful': False,
            'label': 'Normal',
            'confidence': 1.0,
            'warning_message': '',
            'warning_type': 'normal',
        }


# نمونه سراسری تشخیص‌دهنده
spam_detector = SpamDetector()


def check_message(text: str) -> dict:
    """تابع کمکی برای بررسی پیام"""
    return spam_detector.analyze_text(text)
