"""
Inference-time text normalization.

Two separate jobs live here, and the distinction matters:

1. `preprocess_english` / `preprocess_bangla` are ports of the functions in
   Phase 2/Preprocessed Data.ipynb, which produced the `text_clean` column
   every Phase 6 checkpoint was trained on. The extension sends RAW page
   text -- emoji, URLs, hashtags, decorative punctuation and all -- so
   without this the model sees a distribution it never saw during training.

2. `strip_list_prefix` handles something training never had to: real pages
   number their lines. A hand-formatted list renders entries as
   "৭। খাঙ্কিরপোলা", "২১.হাউয়ার পো", "58। বাপের আগে বাল ফালায়". Training
   text had no such prefixes, so the leading numeral is pure
   out-of-distribution noise that shifts the decision. Measured on the
   57-line Bangla slur list from the reported Facebook post, stripping the
   prefix moves hateful recall from 17.5% to 26.3% with no change in
   false-positive rate on bn_test's non-hateful rows -- the single cheapest
   accuracy win available without retraining.

Keep these in sync with the Phase 2 notebook. If the notebook's cleaning
changes and a model is retrained on the result, this file must change too,
or inference silently drifts from training again.
"""
import re
import unicodedata

_URL_RE = re.compile(r"http[s]?://\S+")
_WWW_RE = re.compile(r"www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"\bRT\b")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_ENTITY_RE = re.compile(r"&\w+;")
_WHITESPACE_RE = re.compile(r"\s+")

_EN_KEEP_RE = re.compile(r"[^\w\s.,!?'\"\-:;]")
# Bangla block (U+0980-U+09FF) plus word chars, whitespace, basic punctuation
# and the Bangla danda (।).
_BN_KEEP_RE = re.compile(r"[^ঀ-৿\w\s.,!?'\"\-:;।]")

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "]+",
    flags=re.UNICODE,
)

# A leading list marker: ASCII or Bangla digits (they are freely mixed on real
# pages -- "৫6." and "58।" both appear in the reported post), optionally with
# an interior decimal part, followed by at least one separator. Requiring the
# separator is what stops this from eating a number that is genuinely the
# first word of a sentence ("১০ জন মানুষ" keeps its "১০").
_LIST_PREFIX_RE = re.compile(
    r"^[\s​‌‍]*"  # leading space / zero-width joiners (common in pasted web text)
    r"[0-9০-৯]+"                 # 1  or  ১০৯
    r"(?:[.।৷][0-9০-৯]+)*"       # optional  .১  in "১০৯.১"
    r"\s*[.)\]:।৷-]+\s*"         # separator: . ) ] : । ৷ -
)

# At least one letter (any script) must survive, so "1. 2. 3." style noise
# isn't turned into an empty string and a bare numeric line is left alone.
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def strip_list_prefix(text: str) -> str:
    """
    Remove a leading list-numbering marker ("৭। ", "21.", "১০৯.১ ") if doing
    so still leaves real words behind. Returns `text` unchanged otherwise, so
    this can never empty out a line or destroy a purely numeric one.
    """
    stripped = _LIST_PREFIX_RE.sub("", text, count=1)
    if not _HAS_LETTER_RE.search(stripped):
        return text
    return stripped


def preprocess_english(text: str) -> str:
    """Port of Phase 2's `preprocess_english`. Stop words are deliberately
    kept -- they carry the sarcasm signal."""
    text = _URL_RE.sub("", text)
    text = _WWW_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _RT_RE.sub("", text)
    text = text.replace("&amp;", "and").replace("&lt;", "<").replace("&gt;", ">")
    text = _HTML_ENTITY_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _EN_KEEP_RE.sub(" ", text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def preprocess_bangla(text: str) -> str:
    """Port of Phase 2's `preprocess_bangla`. NFC normalization matters here:
    Bangla conjuncts can arrive decomposed from a web page, and the tokenizer
    treats the decomposed form as different subwords entirely."""
    text = _URL_RE.sub("", text)
    text = _WWW_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _HTML_ENTITY_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _BN_KEEP_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


_BY_LANGUAGE = {"en": preprocess_english, "bn": preprocess_bangla}


def normalize_for_model(text: str, language: str) -> str:
    """
    Full inference-time pipeline: drop page-level list numbering, then apply
    the same cleaning the model was trained on.

    Falls back to whitespace collapsing for an unknown language rather than
    raising -- language validation is the caller's job (main.py rejects
    unsupported languages with a 400 before reaching here), and silently
    passing text through is the safer failure mode for a text cleaner.
    """
    text = strip_list_prefix(text)
    cleaner = _BY_LANGUAGE.get(language)
    if cleaner is None:
        return _WHITESPACE_RE.sub(" ", text).strip()
    return cleaner(text)
