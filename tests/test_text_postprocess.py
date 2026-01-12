import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.text_postprocess import format_allcaps_words, lowercase_long_allcaps_words


def test_lowercase_long_allcaps_words_basic():
    assert lowercase_long_allcaps_words("ONGC ADANIGREENS") == "ONGC adanigreens"


def test_lowercase_long_allcaps_words_with_punctuation():
    assert lowercase_long_allcaps_words("HDFC.NS ADANIPORTS") == "HDFC.NS adaniports"


def test_lowercase_long_allcaps_words_with_digits():
    assert lowercase_long_allcaps_words("NIFTY50 NIFTY") == "NIFTY50 nifty"


def test_lowercase_long_allcaps_words_with_ampersand():
    assert lowercase_long_allcaps_words("S&P RELIANCE") == "S&P reliance"


def test_format_allcaps_words_spaces_short_allcaps():
    assert format_allcaps_words("ONGC will stay ONGC") == "O N G C will stay O N G C"


def test_format_allcaps_words_lowercases_long_allcaps():
    assert format_allcaps_words("ADANIGREENS") == "adanigreens"


def test_format_allcaps_words_ignores_mixed_tokens():
    text = "NIFTY50 RBI. USD/INR HDFC-BANK"
    assert format_allcaps_words(text) == text
