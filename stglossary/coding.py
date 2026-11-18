"""Presence coding of a form in a Chinese output.

Echo rule: an output is coded only up to the first glossary-echo marker, because a model that
copies the glossary into its answer would otherwise score w' through the copy. Markers: ECHO_MARKERS.

Han forms: substring. Latin forms: token boundaries (no ASCII letter or digit on either side);
case-sensitive when the form is all capitals, case-insensitive otherwise.
"""
from __future__ import annotations

import re


ECHO_MARKERS = ("→", "->", "Glossary", "glossary", "词汇表", "术语表", "詞彙表", "術語表")


def translation_part(text: str) -> str:
    """Text before the glossary echo. For an arrow marker the echo line's left term precedes the arrow, so the cut
    moves back to the last sentence end before it, or, if there is none, to the start of the left term."""
    hits = [(i, m) for m in ECHO_MARKERS for i in [text.find(m)] if i >= 0]
    if not hits:
        return text
    i, m = min(hits)
    if m in ("→", "->"):
        ends = [text.rfind(c, 0, i) for c in "。！？.!?\n"]
        e = max(ends)
        if e >= 0:
            return text[: e + 1]
        j = len(text[:i].rstrip())
        while j > 0 and not text[j - 1].isspace():
            j -= 1
        return text[:j]
    return text[:i]


def echoed(text: str) -> bool:
    return any(m in text for m in ECHO_MARKERS)


def has_han(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def is_latin(form: str) -> bool:
    return bool(re.fullmatch(r"[\x00-\x7f]+", form))


def _latin_pattern(form: str) -> tuple[str, int]:
    flags = 0 if (form.upper() == form and any(c.isalpha() for c in form)) else re.I
    return r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])", flags


def contains(text: str, form: str) -> bool:
    if not form:
        return False
    if not is_latin(form):
        return form in text
    pat, flags = _latin_pattern(form)
    return re.search(pat, text, flags) is not None


def spans(text: str, form: str) -> list[tuple[int, int]]:
    """Character spans of every non-overlapping occurrence `contains` would find; [] exactly when it is False."""
    if not form:
        return []
    if not is_latin(form):
        out, i = [], text.find(form)
        while i >= 0:
            out.append((i, i + len(form)))
            i = text.find(form, i + len(form))
        return out
    pat, flags = _latin_pattern(form)
    return [m.span() for m in re.finditer(pat, text, flags)]


def lemma(t: str) -> str:
    """Merge inflectional variants: plural nouns only; acronyms and names are left alone."""
    if not t.islower() and not t.istitle():
        return t.lower()
    t = t.lower()
    if t.endswith("ies") and len(t) > 5:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith(("ss", "us", "is")) and len(t) > 4:
        return t[:-1]
    return t


def code(output: str, w: str, w_wrong: str) -> str:
    output = translation_part(output)
    a, b = contains(output, w_wrong), contains(output, w)
    return {(True, False): "adopted", (True, True): "both", (False, True): "correct", (False, False): "other"}[(a, b)]
