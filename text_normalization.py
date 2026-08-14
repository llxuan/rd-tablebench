"""Separate display-safe repairs from symmetric RD scoring normalization."""

from __future__ import annotations

import re
import unicodedata

CHECKED_TOKEN = "selectionchecked"
HOMOGLYPH_MAP = str.maketrans(
    {
        # Cyrillic
        "а": "a",
        "в": "b",
        "е": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "с": "c",
        "т": "t",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        # Greek
        "α": "a",
        "β": "b",
        "ε": "e",
        "ζ": "z",
        "η": "h",
        "ι": "i",
        "κ": "k",
        "μ": "m",
        "ν": "v",
        "ο": "o",
        "ρ": "p",
        "τ": "t",
        "υ": "y",
        "χ": "x",
    }
)
CROSS_VARIANTS = str.maketrans({character: "x" for character in "✗✘✕✖×╳X"})
DISPLAY_GLYPHS = str.maketrans({"✗": "X", "✘": "X"})


def normalize_display_text(value: object) -> str:
    """Apply only empirically safe changes to released visible output."""
    return str(value or "").translate(DISPLAY_GLYPHS)


def normalize_selection_marks(value: object) -> str:
    """Normalize checkbox semantics without rewriting ordinary prose.

    Checked variants retain a common token. Empty boxes and CU's frequently
    spurious box-cross suffix are optional for scoring. The caller applies this
    function symmetrically to ground truth and prediction.
    """
    text = str(value or "")
    text = re.sub(r"\[\s*[xX✓✔]\s*\]", CHECKED_TOKEN, text)
    text = re.sub(r"\[\s+\]", "", text)
    text = re.sub(r"[✓✔☑]", CHECKED_TOKEN, text)
    text = re.sub(r"[☒☐□◻]", "", text)
    return text


def normalize_scoring_text(value: object) -> str:
    """Apply the versioned symmetric OCR scoring policy."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = normalize_selection_marks(text).lower()
    text = text.translate(HOMOGLYPH_MAP).translate(CROSS_VARIANTS)
    return re.sub(r"[\s-]+", "", text)
