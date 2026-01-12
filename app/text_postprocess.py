import re


_ALLCAPS_LONG_WORD_PATTERN = re.compile(r"\b[A-Z]{5,}\b")
_ALLCAPS_SHORT_WORD_PATTERN = re.compile(r"\b[A-Z]{2,4}\b")


def lowercase_long_allcaps_words(text: str) -> str:
    return _ALLCAPS_LONG_WORD_PATTERN.sub(lambda match: match.group(0).lower(), text)


def space_short_allcaps_words(text: str) -> str:
    return _ALLCAPS_SHORT_WORD_PATTERN.sub(
        lambda match: " ".join(match.group(0)), text
    )


def format_allcaps_words(text: str) -> str:
    return space_short_allcaps_words(lowercase_long_allcaps_words(text))
