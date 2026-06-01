from __future__ import annotations

import re

# Curly/smart quotes → ASCII apostrophe
_APOSTROPHE_VARIANTS = re.compile(r"[''`´]")

_CONTRACTION_FIXES = (
    (r"\bdont\b", "don't"),
    (r"\bwont\b", "won't"),
    (r"\bcant\b", "can't"),
    (r"\bcannot\b", "can't"),
    (r"\bisnt\b", "isn't"),
    (r"\bwasnt\b", "wasn't"),
    (r"\bwouldnt\b", "wouldn't"),
    (r"\bcouldnt\b", "couldn't"),
    (r"\bshouldnt\b", "shouldn't"),
    (r"\bdoesnt\b", "doesn't"),
    (r"\bhasnt\b", "hasn't"),
    (r"\bhavent\b", "haven't"),
    (r"\baren?t\b", "aren't"),
    (r"\bim not\b", "i'm not"),
    (r"\bi am not\b", "i'm not"),
)

_POSITIVE_COME_BACK = re.compile(
    r"\b(?:"
    r"will|can't wait to|cannot wait to|definitely|glad to|love to|plan to|going to|can't wait"
    r")\b[^.]{0,20}\b(?:come|coming)\s+back\b",
    re.IGNORECASE,
)

# Use don'?t / won'?t — \b breaks on apostrophe inside don't/won't
_NEGATOR = r"(?:do not|don'?t|dont|won'?t|wont|will not|wouldn'?t|wouldnt|can'?t|cant|never)"

_NEGATIVE_COME_BACK = re.compile(
    r"(?:"
    rf"\b{_NEGATOR}\s+(?:come|be coming)\s+back\b"
    r"|"
    r"\b(?:never|not)\s+(?:come|coming|going)\s+back\b"
    r"|"
    rf"\b{_NEGATOR}\s+(?:want to\s+)?(?:come back|return)\b"
    r"|"
    rf"\b{_NEGATOR}\s+(?:return|recommend|buy)\b"
    r"|"
    rf"\b{_NEGATOR}\s+want\b"
    r")",
    re.IGNORECASE,
)

_NEGATION_VERB = re.compile(
    rf"\b{_NEGATOR}\s+"
    r"(?:want|like|love|recommend|buy|shop|return|trust|use|order|again)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = _APOSTROPHE_VARIANTS.sub("'", text)
    text = re.sub(r"\s+", " ", text)
    for pattern, replacement in _CONTRACTION_FIXES:
        text = re.sub(pattern, replacement, text)
    return text


def augment_no_apostrophe_variants(text: str) -> list[str]:
    """Training copies so TF-IDF sees both don't and dont."""
    base = clean_text(text)
    variants = [base]
    loose = (
        base.replace("don't", "dont")
        .replace("won't", "wont")
        .replace("can't", "cant")
        .replace("isn't", "isnt")
        .replace("wouldn't", "wouldnt")
        .replace("doesn't", "doesnt")
    )
    if loose != base:
        variants.append(loose)
    return variants


def rule_label(text: str) -> str | None:
    text = clean_text(text)
    if _POSITIVE_COME_BACK.search(text):
        return "positive"
    if _NEGATIVE_COME_BACK.search(text):
        return "negative"
    if _NEGATION_VERB.search(text):
        return "negative"
    return None
