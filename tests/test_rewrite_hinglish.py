import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import rewrite_hinglish


class DummyResult:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class DummyCompletions:
    def __init__(self, outputs: list[str | Exception]):
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []
        self.index = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs[self.index]
        self.index += 1
        if isinstance(output, Exception):
            raise output
        return DummyResult(output)


class DummyChat:
    def __init__(self, outputs: list[str | Exception]):
        self.completions = DummyCompletions(outputs)


class DummyClient:
    def __init__(self, outputs: list[str | Exception]):
        self.chat = DummyChat(outputs)


def test_unchanged_output_triggers_retry():
    block = "Market tone steady with mild optimism."
    outputs = [block, "Market tone steady but slightly upbeat today."]
    client = DummyClient(outputs)

    result = asyncio.run(
        rewrite_hinglish.rewrite_all_blocks(
            [block],
            client=client,
            model_name="test-model",
            temperature=0.6,
        )
    )

    assert result == ["Market tone steady but slightly upbeat today."]
    assert client.chat.completions.index == 2


def test_digit_guard_failure_retries_with_lower_temperature():
    block = "Revenue grew 10% while costs fell."
    outputs = [
        "Revenue grew while costs fell.",
        "Revenue grew 10% while costs fell in this period.",
    ]
    client = DummyClient(outputs)

    asyncio.run(
        rewrite_hinglish.rewrite_all_blocks(
            [block],
            client=client,
            model_name="test-model",
            temperature=0.6,
        )
    )

    assert client.chat.completions.calls[0]["temperature"] == 0.6
    assert client.chat.completions.calls[1]["temperature"] == 0.2
    user_prompt = client.chat.completions.calls[1]["messages"][1]["content"]
    assert "Preserve numbers EXACTLY as written." in user_prompt


def test_fallback_preserves_order_and_length():
    blocks = ["EPS up 5% on strong demand.", "Guidance stays cautious."]
    outputs = [
        "EPS up on strong demand.",
        "EPS up on strong demand.",
        "EPS up on strong demand.",
        "Guidance stays cautious but outlook measured.",
    ]
    client = DummyClient(outputs)

    result = asyncio.run(
        rewrite_hinglish.rewrite_all_blocks(
            blocks,
            client=client,
            model_name="test-model",
            temperature=0.6,
        )
    )

    assert result[0] == "EPS up FIVE% on strong demand."
    assert result[1] == "Guidance stays cautious but outlook measured."
    assert len(result) == len(blocks)
