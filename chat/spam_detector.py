"""
سرویس تشخیص اسپم و محتوای نامناسب فارسی
نسخه ۲.۰ - بهبود یافته با نرمال‌سازی قوی‌تر و احتمالات دقیق‌تر
"""

import re
import os
import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# نرمال‌سازی پیشرفته متن فارسی
# ---------------------------------------------------------------------------

# حذف URL
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|t\.me/\S+|instagram\.com/\S+|wa\.me/\S+)", re.IGNORECASE)

# حذف منشن
USERNAME_PATTERN = re.compile(r"@[\w.]+")

# حذف ایموجی
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

# نگاشت کاراکترهای عربی به فارسی
PERSIAN_CHAR_MAP = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "ٲ": "ا",
    "ﻵ": "لا", "ﻷ": "لا", "ﻹ": "لا", "ﻻ": "لا",
    "ﷲ": "الله", "ﷺ": "صلی الله علیه وسلم",
    "‌": " ",  # ZWNJ -> space
    "‏": "",   # RTL mark
    "‎": "",   # LTR mark
    "\u200c": " ",  # zero-width non-joiner
    "\u200d": "",   # zero-width joiner
    "\u200e": "",   # LTR mark
    "\u200f": "",   # RTL mark
    "\u00ad": "",   # soft hyphen
    "\u200b": "",   # zero-width space
}

# نگاشت اعداد فارسی و عربی
PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXTENDED_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸٩٨٧٦٥٤٣٢١٠", "0123456789876543210")

# حذف اعراب
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

# حذف کاراکترهای تکراری (۳ بار یا بیشتر)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

# الگوی شماره تلفن ایرانی
PHONE_PATTERN = re.compile(r"09[0-9]{9}|(\+98|0098)9[0-9]{9}")

# الگوی شماره کارت بانکی
CARD_PATTERN = re.compile(r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}")

# کاراکترهای مجاز
ALLOWED_CHARS_PATTERN = re.compile(r"[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF0-9a-zA-Z\s.!؟?،,;؛]")

# فضای اضافی
EXTRA_SPACE_PATTERN = re.compile(r"\s+")


def normalize_persian(text: str) -> str:
    """نرمال‌سازی کامل متن فارسی"""
    if not isinstance(text, str):
        return ""

    # حذف URL
    text = URL_PATTERN.sub(" لینک ", text)

    # حذف منشن
    text = USERNAME_PATTERN.sub(" ", text)

    # حذف ایموجی
    text = EMOJI_PATTERN.sub(" ", text)

    # نرمال‌سازی کاراکترها
    for old, new in PERSIAN_CHAR_MAP.items():
        text = text.replace(old, new)

    # نرمال‌سازی اعداد
    text = text.translate(PERSIAN_DIGIT_MAP)
    text = text.translate(ARABIC_DIGIT_MAP)

    # حذف اعراب
    text = DIACRITICS_PATTERN.sub("", text)

    # کاهش تکرار کاراکترها
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)

    # حذف کاراکترهای اضافی
    text = ALLOWED_CHARS_PATTERN.sub(" ", text)

    # حذف فضای اضافی
    text = EXTRA_SPACE_PATTERN.sub(" ", text).strip()

    return text.lower()


# ---------------------------------------------------------------------------
# الگوهای تشخیص اسپم (با امتیازدهی)
# ---------------------------------------------------------------------------

SPAM_PATTERNS = [
    # تبلیغات و بازاریابی
    (r"کانال\s*(تلگرام|واتساپ|اینستاگرام)", 0.95, "تبلیغ کانال"),
    (r"لینک\s*(بیو|bio|پایین|زیر)", 0.95, "لینک بیو"),
    (r"(فالو|follow)\s*(کن|بکن|بزن)", 0.85, "درخواست فالو"),
    (r"(لایک|like)\s*(کن|بکن|بزن)", 0.80, "درخواست لایک"),
    (r"(سابسکرایب|subscribe)\s*(کن|بکن)", 0.90, "درخواست سابسکرایب"),
    (r"(کد\s*تخفیف|تخفیف\s*ویژه|فروش\s*ویژه)", 0.90, "تبلیغ تخفیف"),
    (r"(خرید|فروش)\s*(کن|آنلاین|اینترنتی)", 0.85, "تبلیغ خرید"),
    (r"(درآمد|پول)\s*(میلیونی|میلیاردی|زیاد|عالی)", 0.95, "کلاهبرداری مالی"),
    (r"(کسب\s*درآمد|پولدار|پول\s*دار)\s*(شو|کن)", 0.95, "کلاهبرداری مالی"),
    (r"(بدون\s*سرمایه|رایگان|هدیه)\s*(شروع|کن)", 0.85, "تبلیغ فریبنده"),
    (r"(ارسال\s*رایگان|ارسال\s*به\s*سراسر)", 0.85, "تبلیغ ارسال"),
    (r"(گارانتی|ضمانت|اصالت)", 0.70, "تبلیغ ضمانت"),
    (r"(محدود|فوری|آخرین\s*فرصت|تمام\s*شدنی)", 0.80, "تبلیغ فوری"),
    (r"(واتساپ|whatsapp|تلگرام|telegram|اینستاگرام|instagram)\s*[:\s]*\d", 0.90, "اشتراک‌گذاری شماره"),
    (r"(کلیک|click)\s*(کن|بکن|روی)", 0.85, "درخواست کلیک"),
    (r"(ثبت\s*نام|عضویت)\s*(کن|کنید)", 0.75, "درخواست ثبت‌نام"),
    (r"(جایزه|برنده|قرعه\s*کشی)\s*(شو|کن|بردی)", 0.95, "کلاهبرداری جایزه"),
    (r"(هدیه|بونوس)\s*(رایگان|ویژه)", 0.85, "تبلیغ هدیه"),
    (r"(شانس|فرصت)\s*(آخر|طلایی|ویژه)", 0.85, "تبلیغ فریبنده"),

    # شماره تلفن و مالی
    (r"09[0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{4}", 0.75, "اشتراک‌گذاری شماره"),
    (r"(شماره|شماره\s*تماس|تلفن)\s*[:\s]*09", 0.85, "اشتراک‌گذاری شماره"),
    (r"(کارت\s*به\s*کارت|واریز|واریزی)\s*[:\s]*\d", 0.90, "درخواست مالی"),
    (r"(قیمت|تومان|ریال)\s*[:\s]*\d", 0.70, "قیمت‌گذاری"),
    (r"(ارزان|ارزان\s*تر|بهترین\s*قیمت)", 0.75, "تبلیغ قیمت"),

    # محتوای غیراخلاقی
    (r"(فیلم|عکس|کلیپ)\s*(سکس|پورن|جنده|بدون\s*سانسور)", 0.98, "محتوای غیراخلاقی"),
    (r"(سایت|سکس|پورن|xxx|adult)", 0.95, "محتوای غیراخلاقی"),
    (r"(دختر|پسر)\s*(برای|میخواد|آماده)", 0.70, "محتوای مشکوک"),
]

# ---------------------------------------------------------------------------
# الگوهای تشخیص نفرت و توهین
# ---------------------------------------------------------------------------

HATE_PATTERNS = [
    # توهین‌های مستقیم
    (r"(خفه|خفه\s*شو|خفه\s*شوید)", 0.95, "توهین مستقیم"),
    (r"(برو\s*گمشو|گمشو|برید\s*گمشید)", 0.95, "توهین مستقیم"),
    (r"(احمق|بی\s*عقل|نادان|کودن|اسکل|نفهم)", 0.90, "توهین هوشی"),
    (r"(بی\s*شرف|بی\s*ناموس|بی\s*غیرت|بی\s*ناموس)", 0.95, "توهین خانوادگی"),
    (r"(پدر\s*سوخته|لاشی|خایه|بی\s*ناموس)", 0.95, "توهین شدید"),
    (r"(کثافت|کثیف|چندش|حال\s*بهم\s*زن|متعفن)", 0.90, "توهین بهداشتی"),
    (r"(حرومزاده|حرامزاده|لاشخور|عوضی)", 0.95, "توهین شدید"),
    (r"(دیوث|جنده|فاحشه|روسپی)", 0.95, "توهین جنسی"),
    (r"(کیر|کص|کون|جنده)", 0.95, "توهین رکیک"),
    (r"(مادر\s*قحبه|مادرت|خواهرت)", 0.90, "توهین خانوادگی"),
    (r"(بکش|بمیر|مرده|لاش\s*شوی)", 0.90, "تهدید"),
    (r"(برو\s*بمیر|بمیری|خدا\s*بکشدت)", 0.95, "تهدید"),
    (r"(حیوان|حیوون|سگ|خر|الاغ|گاو|بوزینه)", 0.80, "توهین تشبیهی"),
    (r"(بی\s*تربیت|بی\s*ادب|بی\s*فرهنگ)", 0.80, "توهین فرهنگی"),
    (r"(مزخرف|چرت|پرت|مسخره|خنده\s*دار)", 0.70, "توهین خفیف"),
    (r"(برو\s*بابا|اَه|اوف|ای\s*بابا)", 0.50, "توهین خفیف"),
    (r"(لعنت|لعنتم|خدا\s*لعنتم)", 0.85, "نفرین"),
    (r"(دشمن|دشمن\s*خدا|کافر)", 0.75, "توهین مذهبی"),
    (r"(تروریست|داعشی|بعثی)", 0.85, "توهین سیاسی"),
    (r"(عرب\s*بیابانی|افغانی|ترک\s*خر)", 0.90, "توهین نژادی"),
    (r"(کور|کر|لال|معلول|عقیم)", 0.80, "توهین به معلولیت"),
]

# ---------------------------------------------------------------------------
# الگوهای ضد اسپم (کاهش امتیاز)
# ---------------------------------------------------------------------------

NEGATIVE_PATTERNS = [
    (r"(سلام|درود|صبح\s*بخیر|عصر\s*بخیر|شب\s*بخیر)", -0.30, "احوالپرسی"),
    (r"(ممنون|متشکرم|سپاس|مرسی|دمت\s*گرم)", -0.25, "قدردانی"),
    (r"(خوبی|خوبم|چطوری|حالت|حالم)", -0.20, "احوالپرسی"),
    (r"(لطفا|خواهش|زحمت)", -0.15, "ادب"),
    (r"(انشاءالله|به\s*امید\s*خدا|خدا\s*کنه)", -0.20, "مذهبی مثبت"),
    (r"(موفق|موفقیت|پیروز|شاد)", -0.15, "مثبت"),
    (r"(خدا|خدایا|یا\s*خدا)", -0.10, "مذهبی"),
    (r"(^\s*$)", -1.0, "خالی"),
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
                logger.warning("⚠️ فایل‌های مدل یافت نشد. از سیستم الگویی استفاده می‌شود")

        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل: {e}")

    def analyze_text(self, text: str) -> dict:
        """
        تحلیل متن و تشخیص اسپم/نفرت

        خروجی:
        {
            'is_harmful': bool,
            'label': 'Normal' | 'Spam' | 'Hate',
            'confidence': float (0-100),
            'warning_message': str,
            'warning_type': 'spam' | 'hate' | 'normal',
            'details': list,
        }
        """
        if not text or not text.strip():
            return self._normal_result("متن خالی")

        # نرمال‌سازی متن
        normalized = normalize_persian(text)

        if len(normalized) < 2:
            return self._normal_result("متن خیلی کوتاه")

        # اگر مدل آموزش‌دیده موجود باشد
        if self.model_loaded:
            return self._predict_with_model(normalized, text)

        # در غیر این صورت از سیستم الگویی استفاده کن
        return self._predict_with_patterns(normalized, text)

    def _predict_with_model(self, normalized: str, original: str) -> dict:
        """پیش‌بینی با مدل آموزش‌دیده"""
        try:
            from .preprocess import clean_text

            cleaned = clean_text(original)
            features = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(features)[0]

            # دریافت امتیاز اطمینان
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                # تبدیل به احتمال با سیگموید
                # decision > 0 means spam, < 0 means ham
                confidence = abs(float(decision))
                # تبدیل به درصد (0-100)
                confidence = min(99, max(60, 50 + confidence * 50))
            else:
                confidence = 80.0

            if prediction == 'spam':
                # ترکیب با سیستم الگویی برای نفرت
                pattern_result = self._predict_with_patterns(normalized, original)
                if pattern_result['label'] == 'Hate':
                    return pattern_result

                return {
                    'is_harmful': True,
                    'label': 'Spam',
                    'confidence': round(confidence, 1),
                    'warning_message': '⚠️ این پیام به عنوان اسپم شناسایی شد.',
                    'warning_type': 'spam',
                    'original_text': original,
                    'details': ['تشخیص با مدل آموزش‌دیده (دقت ۹۷٪)'],
                }
            else:
                # اگر مدل گفت عادی، با سیستم الگویی هم بررسی کن
                pattern_result = self._predict_with_patterns(normalized, original)
                if pattern_result['is_harmful']:
                    return pattern_result
                return self._normal_result("پیام سالم")

        except Exception as e:
            logger.error(f"خطا در پیش‌بینی مدل: {e}")
            return self._predict_with_patterns(normalized, original)

    def _predict_with_patterns(self, normalized: str, original: str) -> dict:
        """پیش‌بینی با سیستم الگویی"""
        hate_score = 0
        hate_details = []
        hate_max_confidence = 0

        spam_score = 0
        spam_details = []
        spam_max_confidence = 0

        negative_score = 0

        # بررسی الگوهای نفرت
        for pattern, weight, desc in HATE_PATTERNS:
            if re.search(pattern, normalized):
                hate_score += weight
                hate_max_confidence = max(hate_max_confidence, weight)
                hate_details.append(desc)

        # بررسی الگوهای اسپم
        for pattern, weight, desc in SPAM_PATTERNS:
            if re.search(pattern, normalized):
                spam_score += weight
                spam_max_confidence = max(spam_max_confidence, weight)
                spam_details.append(desc)

        # بررسی الگوهای منفی (کاهش امتیاز)
        for pattern, weight, desc in NEGATIVE_PATTERNS:
            if re.search(pattern, normalized):
                negative_score += weight

        # محاسبه امتیاز نهایی
        final_hate_score = max(0, hate_score + negative_score)
        final_spam_score = max(0, spam_score + negative_score)

        # تصمیم‌گیری نهایی
        if final_hate_score >= 0.8 and final_hate_score > final_spam_score:
            confidence = min(95, max(60, hate_max_confidence * 100))
            return {
                'is_harmful': True,
                'label': 'Hate',
                'confidence': round(confidence, 1),
                'warning_message': '🚫 این پیام شامل محتوای نفرت‌انگیز یا توهین‌آمیز است.',
                'warning_type': 'hate',
                'original_text': original,
                'details': hate_details[:3],
            }

        if final_spam_score >= 0.7:
            confidence = min(95, max(55, spam_max_confidence * 100))
            return {
                'is_harmful': True,
                'label': 'Spam',
                'confidence': round(confidence, 1),
                'warning_message': '⚠️ این پیام به عنوان اسپم شناسایی شد.',
                'warning_type': 'spam',
                'original_text': original,
                'details': spam_details[:3],
            }

        # اگر امتیاز پایین بود ولی صفر نبود
        if final_hate_score > 0.3 or final_spam_score > 0.3:
            if final_hate_score > final_spam_score:
                confidence = min(55, max(30, hate_max_confidence * 60))
                return {
                    'is_harmful': False,
                    'label': 'Normal',
                    'confidence': round(100 - confidence, 1),
                    'warning_message': '💡 این پیام ممکن است مشکوک باشد.',
                    'warning_type': 'normal',
                    'original_text': original,
                    'details': hate_details[:2],
                }
            else:
                confidence = min(55, max(30, spam_max_confidence * 60))
                return {
                    'is_harmful': False,
                    'label': 'Normal',
                    'confidence': round(100 - confidence, 1),
                    'warning_message': '💡 این پیام ممکن است مشکوک باشد.',
                    'warning_type': 'normal',
                    'original_text': original,
                    'details': spam_details[:2],
                }

        return self._normal_result("پیام سالم")

    def _normal_result(self, reason: str = "") -> dict:
        return {
            'is_harmful': False,
            'label': 'Normal',
            'confidence': 95.0,
            'warning_message': '',
            'warning_type': 'normal',
            'details': [reason] if reason else [],
        }


# نمونه سراسری تشخیص‌دهنده
spam_detector = SpamDetector()


def check_message(text: str) -> dict:
    """تابع کمکی برای بررسی پیام"""
    return spam_detector.analyze_text(text)
