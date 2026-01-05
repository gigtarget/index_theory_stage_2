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


def test_slide_one_hook_and_word_cap():
    script = (
        "Aaj ka plan kya hai? Quick take: revenue steady, costs trimmed, mood upbeat. "
        "Expect crisp pointers and grounded numbers to set the vibe. Next, we'll look at margin trends and what shifted."
    )
    assert script_generator._word_count(script) <= 40
    assert script_generator.slide_one_has_hook(script) is True


def test_last_slide_requires_cta_and_risk():
    script = (
        "Recap the key signals and avoid over-trading. We saw banks hold steady while volumes cooled. "
        "Volatility softer but not gone. If this helped, like and subscribe, and trade light to protect capital."
    )
    is_valid, errors = script_generator.validate_script_rules(
        script, is_last=True, next_intent="", is_first=False, is_low_context=False
    )
    assert is_valid is True
    assert errors == []


def test_middle_slide_needs_transition_and_no_cta():
    script = (
        "Growth slowed but services held up. Costs are easing, giving small relief. Next, we'll look at margin shifts across segments."
    )
    is_valid, errors = script_generator.validate_script_rules(
        script,
        is_last=False,
        next_intent="margin shifts across segments",
        is_first=False,
        is_low_context=False,
    )
    assert is_valid is True
    assert errors == []


def test_middle_low_context_slide_word_cap_and_greeting_ban():
    script = (
        "Quick setup on global headwinds, market thoda tight hai. Expect earnings update aage. Next, we'll look at margin updates."
    )
    is_valid, errors = script_generator.validate_script_rules(
        script,
        is_last=False,
        next_intent="margin updates",
        is_first=False,
        is_low_context=True,
    )
    assert script_generator._word_count(script) <= 45
    assert is_valid is True
    assert "banned greeting" not in errors


def test_slide_one_allows_welcome_and_word_cap():
    script = (
        "Welcome to the weekly outlook. Aaj ka plan simple hai: key charts dekhenge aur clear takeaways milenge. Next, we'll look at market movers."
    )
    is_valid, errors = script_generator.validate_script_rules(
        script,
        is_last=False,
        next_intent="market movers",
        is_first=True,
        is_low_context=False,
    )
    assert script_generator._word_count(script) <= 40
    assert is_valid is True
    assert errors == []


def test_last_slide_has_cta_and_no_next_line():
    script = (
        "Quick recap aur guardrails dhyan rakho. Gains ko lock karo, over-trading avoid karo. If this helped, like aur subscribe karna, aur har trade me risk yaad rakho."
    )
    is_valid, errors = script_generator.validate_script_rules(
        script,
        is_last=True,
        next_intent="",
        is_first=False,
        is_low_context=False,
    )
    assert is_valid is True
    assert "last slide has next transition" not in errors
