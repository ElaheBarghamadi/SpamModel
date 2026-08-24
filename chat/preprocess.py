"""
preprocess.py
-------------
Text cleaning pipeline for Persian (Farsi) social media comments.
نسخه بهبود یافته برای استفاده در اپلیکیشن چت
"""

import re

# الگوهای پیش‌پردازش
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
