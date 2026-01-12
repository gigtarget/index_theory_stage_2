import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.num_to_words import convert_numerals_to_words, has_disallowed_digits


def test_convert_simple_integer():
    assert convert_numerals_to_words("50") == "FIFTY"


def test_convert_comma_integer():
    assert (
        convert_numerals_to_words("3,234")
        == "THREE THOUSAND TWO HUNDRED AND THIRTY FOUR"
    )


def test_protected_token_nifty():
    assert convert_numerals_to_words("NIFTY 50 was steady") == "NIFTY 50 was steady"


def test_has_disallowed_digits_protected_token():
    assert has_disallowed_digits("NIFTY 50") is False
