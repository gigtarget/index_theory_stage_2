import re


_ALLCAPS_LONG_WORD_PATTERN = re.compile(r"\b[A-Z]{5,}\b")


def lowercase_long_allcaps_words(text: str) -> str:
    return _ALLCAPS_LONG_WORD_PATTERN.sub(lambda match: match.group(0).lower(), text)
