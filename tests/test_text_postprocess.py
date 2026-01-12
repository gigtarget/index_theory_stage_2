import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.text_postprocess import lowercase_long_allcaps_words


def test_lowercase_long_allcaps_words_basic():
    assert lowercase_long_allcaps_words("ONGC ADANIGREENS") == "ONGC adanigreens"


def test_lowercase_long_allcaps_words_with_punctuation():
    assert lowercase_long_allcaps_words("HDFC.NS ADANIPORTS") == "HDFC.NS adaniports"


def test_lowercase_long_allcaps_words_with_digits():
    assert lowercase_long_allcaps_words("NIFTY50 NIFTY") == "NIFTY50 nifty"


def test_lowercase_long_allcaps_words_with_ampersand():
    assert lowercase_long_allcaps_words("S&P RELIANCE") == "S&P reliance"
