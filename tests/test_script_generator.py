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
        "Welcome to today's report. We're diving into revenue growth with simple insights."
        " Key highlights stay crisp and easy to follow."
        " You'll get clear takeaways in minutes."
        " Next, we'll look at revenue growth and what's changing."
    )
    assert script_generator.script_is_plain_narration(script) is True


def test_script_has_no_banned_labels_rejects_headers():
    script = (
        "Slide 2 Script\n"
        "Hook: opening line\n"
        "Key points: detail one. detail two.\n"
        "Takeaway: summary line.\n"
        "Transition: Next, we'll look at revenue growth."
    )
    assert script_generator.script_has_no_banned_labels(script) is False


def test_find_transition_sentence_detects_required_prefix():
    script = "Great intro. Clear facts follow quickly. Next, we'll look at market share changes across regions."
    transition = script_generator.find_transition_sentence(script)
    assert transition.startswith("Next, we'll look at") or transition.startswith("Next, we’ll look at")


def test_transition_mentions_intent():
    intent = "market share changes"
    script = "Strong hook. Quick points share value. Next, we'll look at market share changes and compare performance."
    assert script_generator.transition_mentions_intent(script, intent) is True


def test_slide_one_word_limit_cap():
    text = " ".join(["word"] * 120)
    truncated, applied = script_generator._enforce_word_limit(text, max_words=70)
    assert applied is True
    assert len(truncated.split()) == 70
