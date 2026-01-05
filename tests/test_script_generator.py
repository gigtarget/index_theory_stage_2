import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import script_generator


def test_enforce_word_limit_truncates():
    text = " ".join(["word"] * 120)
    truncated, applied = script_generator._enforce_word_limit(text, max_words=95)
    assert applied is True
    assert len(truncated.split()) == 95


def test_script_has_required_sections():
    script = (
        "Hook: opening line\n"
        "Key points: detail one. detail two.\n"
        "Takeaway: summary line.\n"
        "Transition: Next, we'll look at revenue growth."
    )
    assert script_generator.script_has_required_sections(script) is True


def test_transition_mentions_intent():
    intent = "revenue growth"
    script = (
        "Hook: intro\n"
        "Key points: first. second.\n"
        "Takeaway: wrap.\n"
        "Transition: Next, we'll look at revenue growth and compare performance."
    )
    assert script_generator.transition_mentions_intent(script, intent) is True
