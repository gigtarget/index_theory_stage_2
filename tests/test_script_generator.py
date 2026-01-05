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


def test_script_is_plain_narration():
    script = (
        "Bold hook to draw you in."
        " Two crisp points land clearly."
        " Final thought that sticks."
        " Next, we'll look at revenue growth and what's changing."
    )
    assert script_generator.script_is_plain_narration(script) is True


def test_script_is_plain_narration_rejects_labels():
    script = (
        "Slide 2 Script\n"
        "Hook: opening line\n"
        "Key points: detail one. detail two.\n"
        "Takeaway: summary line.\n"
        "Transition: Next, we'll look at revenue growth."
    )
    assert script_generator.script_is_plain_narration(script) is False


def test_transition_mentions_intent():
    intent = "revenue growth"
    script = "Strong hook. Quick points share value. Next, we'll look at revenue growth and compare performance."
    assert script_generator.transition_mentions_intent(script, intent) is True
