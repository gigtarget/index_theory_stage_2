import re

LATIN_TO_DEVANAGARI = {
    "a": "अ",
    "b": "ब",
    "c": "क",
    "d": "द",
    "e": "ए",
    "f": "फ",
    "g": "ग",
    "h": "ह",
    "i": "इ",
    "j": "ज",
    "k": "क",
    "l": "ल",
    "m": "म",
    "n": "न",
    "o": "ओ",
    "p": "प",
    "q": "क",
    "r": "र",
    "s": "स",
    "t": "त",
    "u": "उ",
    "v": "व",
    "w": "व",
    "x": "क्ष",
    "y": "य",
    "z": "ज",
}

PROTECTED_SPANS = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+|\b[\w.+-]+@[\w.-]+\.\w+\b|@\w+)"
)


def _convert_segment(segment: str) -> str:
    return "".join(LATIN_TO_DEVANAGARI.get(char.lower(), char) for char in segment)


def transliterate_to_devanagari(text: str) -> str:
    if not text:
        return ""

    parts: list[str] = []
    cursor = 0
    for match in PROTECTED_SPANS.finditer(text):
        start, end = match.span()
        if start > cursor:
            parts.append(_convert_segment(text[cursor:start]))
        parts.append(text[start:end])
        cursor = end
    if cursor < len(text):
        parts.append(_convert_segment(text[cursor:]))
    return "".join(parts)
