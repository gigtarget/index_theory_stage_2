import os

from app import script_generator


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class DummyCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def create(self, *args, **kwargs):
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = ""
        return DummyResponse(content)


class DummyChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = DummyCompletions(responses)


class DummyClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = DummyChat(responses)


def test_humanize_dedupes_banned_phrases(monkeypatch):
    monkeypatch.setenv("HINDI_DEVANAGARI", "0")
    full_script = (
        "Let's start with Nifty 25000.\n"
        "Let's start with Nifty 25000 again.\n"
        "Quick recap: FII 1200.\n"
        "Quick recap: FII 1200."
    )
    client = DummyClient([full_script])
    output = script_generator.humanize_full_script(full_script, client, "dummy")
    for phrase in script_generator.BANNED_REPETITIVE_PHRASES:
        assert output.lower().count(phrase.lower()) <= 1


def test_humanize_preserves_digits_guard(monkeypatch):
    monkeypatch.setenv("HINDI_DEVANAGARI", "0")
    full_script = "Nifty 25000. Bank Nifty 52000. Up 0.5%."
    client = DummyClient(["Nifty moved higher with no numbers mentioned."])
    output = script_generator.humanize_full_script(full_script, client, "dummy")
    assert output == full_script
