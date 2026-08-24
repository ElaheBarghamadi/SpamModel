"""
preprocess.py
-------------
Text cleaning pipeline for Persian (Farsi) social media comments.

The goal of this module is to normalize noisy, informal text (Instagram
comments) into a cleaner form that is easier for a TF-IDF + classical ML
model to learn from, WITHOUT destroying the meaning of the text.

Design choices:
    - Stop words are NOT removed (per project requirements).
    - We normalize common Arabic/Persian character variants
      (e.g. "ي" -> "ی", "ك" -> "ک") since both forms appear in the wild.
    - We remove things that add noise but carry little classification
      signal on their own: URLs, @usernames, emojis, extra whitespace,
      and most punctuation — while keeping Persian letters and digits.

v1.1 additions (filter-evasion / obfuscation handling):
    People trying to dodge automated moderation often disguise offensive
    or spam words by:
      1. Stretching letters for emphasis/spam ("سلاااام" instead of "سلام").
      2. Splitting a word into single letters joined by separators like
         dots, stars, underscores, or hyphens ("ک.ی.ر", "ک_ی_ر", "ک*ی*ر").
      3. Fully masking a word with symbols ("****", "####") — the word
         itself is unrecoverable, but the presence of a self-censored
         word is itself a strong signal.
    This module now normalizes (1) and (2) back into a single clean word,
    and replaces (3) with a placeholder token so the classifier can still
    learn from the fact that *something* was censored there.
"""

import re

# ---------------------------------------------------------------------------
# Precompiled regular expressions (compiling once is faster than compiling
# the same pattern on every function call).
# ---------------------------------------------------------------------------

# Matches http(s):// links and bare www. links.
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")

# Matches @username mentions.
USERNAME_PATTERN = re.compile(r"@\w+")

# Matches most emoji / pictograph unicode ranges.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flags)
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "\U0001F000-\U0001F0FF"
    "]+",
    flags=re.UNICODE,
)

# Collapses any run of whitespace (spaces, tabs, newlines) into one space.
EXTRA_SPACE_PATTERN = re.compile(r"\s+")

# Characters to normalize: Arabic forms -> Persian forms (and a few other
# common look-alike variants seen in real Instagram comments).
PERSIAN_CHAR_MAP = {
    "ي": "ی",  # Arabic Yeh -> Persian Yeh
    "ى": "ی",  # Alef Maksura -> Persian Yeh
    "ك": "ک",  # Arabic Kaf -> Persian Kaf
    "ة": "ه",  # Arabic Teh Marbuta -> Persian Heh
    "ۀ": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
    "ٲ": "ا",
    "ﻻ": "لا",
    "ﷲ": "الله",
    "‌": " ",  # zero-width non-joiner -> normal space
    "‏": "",  # right-to-left mark -> remove
    "‎": "",  # left-to-right mark -> remove
}

# Persian and Arabic-Indic digits -> plain Latin digits, so numbers
# (phone numbers, prices, etc.) are represented consistently regardless
# of which digit script the commenter used.
PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Arabic diacritics (tashkeel/harakat) — decorative marks that add noise
# without changing the base letters.
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

# Three or more repeated identical Persian/Arabic letters in a row are
# almost always emphasis/stretching ("سلاااام"), not intentional
# spelling — collapse them to a single letter. Restricted to letters
# (not digits) so phone numbers / prices with repeated digits are untouched.
LETTER_ELONGATION_PATTERN = re.compile(r"([\u0600-\u06FF])\1{2,}")

# A single Persian/Arabic letter, used to build the de-obfuscation patterns.
_PERSIAN_LETTER = r"[\u0600-\u06FF]"

# Matches a word that has been split into single letters joined by
# unambiguous "separator" symbols used to dodge keyword filters, e.g.
# "ک.ی.ر", "ک_ی_ر", "ک*ی*ر", "ک-ی-ر". The lookaround ensures we only
# match when every "letter" is isolated (not part of a bigger word), so
# normal text is left untouched.
SEPARATOR_OBFUSCATION_PATTERN = re.compile(
    rf"(?<!{_PERSIAN_LETTER})(?:{_PERSIAN_LETTER}[.*_\-]+){{1,}}{_PERSIAN_LETTER}(?!{_PERSIAN_LETTER})"
)

# Same idea, but for single letters separated by plain spaces
# ("ک ی ر"). This requires at least 3 letters total (2+ separators) to
# reduce the chance of merging legitimate short/one-letter words that
# happen to sit next to each other in normal sentences.
SPACED_OBFUSCATION_PATTERN = re.compile(
    rf"(?<!{_PERSIAN_LETTER})(?:{_PERSIAN_LETTER}\s+){{2,}}{_PERSIAN_LETTER}(?!{_PERSIAN_LETTER})"
)

# A "word" made entirely of censor symbols (no letters at all), e.g.
# "****", "####", "----". These can't be reconstructed, but their
# presence is itself a signal, so they're replaced with a placeholder
# token instead of being silently deleted.
FULLY_CENSORED_TOKEN_PATTERN = re.compile(r"(?<!\S)[*#@$%_\-]{2,}(?!\S)")
CENSORED_TOKEN_PLACEHOLDER = "سانسور"

# Keep Persian letters, digits (Persian + Latin), whitespace, and basic
# sentence punctuation that carries some meaning (. ! ?). Everything else
# (extra punctuation, symbols) is removed.
ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF0-9a-zA-Z\s.!؟?]"
)


def remove_urls(text: str) -> str:
    """Remove http(s)/www links from the text."""
    return URL_PATTERN.sub(" ", text)


def remove_usernames(text: str) -> str:
    """Remove @username mentions from the text."""
    return USERNAME_PATTERN.sub(" ", text)


def remove_emojis(text: str) -> str:
    """Remove emoji and pictograph characters from the text."""
    return EMOJI_PATTERN.sub(" ", text)


def normalize_persian_characters(text: str) -> str:
    """Normalize Arabic character variants to their Persian equivalents."""
    for arabic_char, persian_char in PERSIAN_CHAR_MAP.items():
        text = text.replace(arabic_char, persian_char)
    return text


def normalize_persian_digits(text: str) -> str:
    """Convert Persian and Arabic-Indic digits to plain Latin digits."""
    text = text.translate(PERSIAN_DIGIT_MAP)
    text = text.translate(ARABIC_DIGIT_MAP)
    return text


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel/harakat)."""
    return DIACRITICS_PATTERN.sub("", text)


def reduce_letter_elongation(text: str) -> str:
    """
    Collapse letters repeated 3+ times in a row down to one occurrence.
    Handles emphasis/stretching such as "سلاااام" -> "سلام" or
    "خییییلی" -> "خیلی", which is common in spam and emotionally
    charged (hateful) comments.
    """
    return LETTER_ELONGATION_PATTERN.sub(r"\1", text)


def merge_obfuscated_letters(text: str) -> str:
    """
    Reassemble words that were deliberately split into single letters to
    dodge keyword-based filters, e.g.:
        "ک.ی.ر"  -> "کیر"
        "ک_ی_ر"  -> "کیر"
        "ک*ی*ر"  -> "کیر"
        "ک ی ر"  -> "کیر"

    This targets a common filter-evasion trick without touching normal
    Persian sentences (short, distinct words are left alone — see the
    pattern definitions above for the exact safeguards).
    """
    def _strip_separators(match: re.Match) -> str:
        return re.sub(r"[\s.*_\-]+", "", match.group(0))

    text = SEPARATOR_OBFUSCATION_PATTERN.sub(_strip_separators, text)
    text = SPACED_OBFUSCATION_PATTERN.sub(_strip_separators, text)
    return text


def mark_fully_censored_tokens(text: str) -> str:
    """
    Replace tokens made entirely of censor symbols (e.g. "****", "####")
    with a placeholder word. The original word can't be recovered, but
    the fact that the author self-censored something is a useful signal
    for the classifier to learn from, so it's kept rather than deleted.
    """
    return FULLY_CENSORED_TOKEN_PATTERN.sub(CENSORED_TOKEN_PLACEHOLDER, text)


def remove_unnecessary_punctuation(text: str) -> str:
    """
    Remove punctuation/symbols that don't help classification, while
    keeping Persian letters, digits, whitespace, and basic sentence
    punctuation (. ! ? ؟) so the text stays readable.
    """
    return ALLOWED_CHARS_PATTERN.sub(" ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space and trim."""
    return EXTRA_SPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """
    Run the full preprocessing pipeline on a single piece of text.

    Order matters:
      1. Strip URLs/usernames/emojis first, before anything else, so
         leftovers from links or mentions don't get misread as
         punctuation noise or obfuscation patterns later on.
      2. Normalize characters and digits before the obfuscation/censor
         steps, so those steps see a consistent character set.
      3. Handle de-obfuscation (elongation, split letters, censor
         tokens) BEFORE removing punctuation — the punctuation removal
         step would otherwise destroy the separators these steps rely on.
      4. Remove remaining unnecessary punctuation and collapse
         whitespace last, as a final cleanup pass.
    """
    if not isinstance(text, str):
        return ""

    text = remove_urls(text)
    text = remove_usernames(text)
    text = remove_emojis(text)

    text = normalize_persian_characters(text)
    text = normalize_persian_digits(text)
    text = remove_diacritics(text)

    text = reduce_letter_elongation(text)
    text = merge_obfuscated_letters(text)
    text = mark_fully_censored_tokens(text)

    text = remove_unnecessary_punctuation(text)
    text = remove_extra_spaces(text)

    return text


def clean_text_series(texts):
    """
    Apply clean_text() to an iterable/Series of texts.
    Convenience wrapper used by both train.py and predict.py so the
    exact same cleaning logic is applied at train and inference time.
    """
    return [clean_text(t) for t in texts]
