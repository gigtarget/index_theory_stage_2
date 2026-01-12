import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

PROTECTED_PATTERNS = [
    re.compile(r"\bNIFTY\s+50\b", re.IGNORECASE),
    re.compile(r"\bBANKNIFTY\b", re.IGNORECASE),
    re.compile(r"\bSENSEX\b", re.IGNORECASE),
    re.compile(r"\bVIX\b", re.IGNORECASE),
    re.compile(r"\bS&P\s+500\b", re.IGNORECASE),
    re.compile(
        r"\b(?=[A-Z0-9&_-]{2,}(?:\.[A-Z]{1,5})?\b)(?=[A-Z0-9&_-]*[A-Z])[A-Z0-9&_-]{2,}(?:\.[A-Z]{1,5})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\.[NB]S\b", re.IGNORECASE),
    re.compile(r"\.[NB]O\b", re.IGNORECASE),
]

NUMBER_PATTERN = re.compile(
    r"\b-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b-?\d+(?:\.\d+)?\b"
)

_UNITS = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
]

_TENS = [
    "",
    "",
    "TWENTY",
    "THIRTY",
    "FORTY",
    "FIFTY",
    "SIXTY",
    "SEVENTY",
    "EIGHTY",
    "NINETY",
]

_DIGIT_WORDS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}


def _protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    replacements: Dict[str, str] = {}
    counter = 0

    for pattern in PROTECTED_PATTERNS:
        def _repl(match: re.Match[str]) -> str:
            nonlocal counter
            placeholder = f"__PROTECTED_{counter}__"
            replacements[placeholder] = match.group(0)
            counter += 1
            return placeholder

        text = pattern.sub(_repl, text)

    return text, replacements


def _restore_tokens(text: str, replacements: Dict[str, str]) -> str:
    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    return text


def _convert_hundreds(number: int) -> str:
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        if remainder == 0:
            return _TENS[tens]
        return f"{_TENS[tens]} {_UNITS[remainder]}"

    hundreds, remainder = divmod(number, 100)
    words = f"{_UNITS[hundreds]} HUNDRED"
    if remainder:
        words = f"{words} AND {_convert_hundreds(remainder)}"
    return words


def _number_to_words(number: int) -> str:
    if number < 1000:
        return _convert_hundreds(number)
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1000)
        words = f"{_convert_hundreds(thousands)} THOUSAND"
        if remainder:
            words = f"{words} {_convert_hundreds(remainder)}"
        return words
    if number < 1_000_000_000:
        millions, remainder = divmod(number, 1_000_000)
        words = f"{_convert_hundreds(millions)} MILLION"
        if remainder:
            words = f"{words} {_number_to_words(remainder)}"
        return words

    raise ValueError("Number out of supported range")


def _convert_number_string(token: str) -> str | None:
    negative = token.startswith("-")
    core = token[1:] if negative else token
    if core.count(".") > 1:
        return None

    if "." in core:
        integer_part, decimal_part = core.split(".", 1)
    else:
        integer_part, decimal_part = core, None

    integer_digits = integer_part.replace(",", "")
    if not integer_digits.isdigit():
        return None

    number = int(integer_digits)
    if number > 999_999_999:
        return None

    try:
        words = _number_to_words(number)
    except ValueError:
        return None

    if decimal_part is not None:
        if not decimal_part.isdigit():
            return None
        decimal_words = " ".join(_DIGIT_WORDS[digit] for digit in decimal_part)
        words = f"{words} POINT {decimal_words}"

    if negative:
        words = f"MINUS {words}"
    return words


def has_disallowed_digits(text: str) -> bool:
    protected_text, _ = _protect_tokens(text)
    return bool(re.search(r"\d", protected_text))


def convert_numerals_to_words(text: str) -> str:
    protected_text, replacements = _protect_tokens(text)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        converted = _convert_number_string(token)
        if converted is None:
            logger.warning("Unable to convert number token '%s'; leaving as-is.", token)
            return token
        return converted

    converted_text = NUMBER_PATTERN.sub(_replace, protected_text)
    return _restore_tokens(converted_text, replacements)
