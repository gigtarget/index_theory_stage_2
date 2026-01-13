import logging

from app.script_generator import _build_client, _get_model_name

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a strict transliteration engine. Convert Latin alphabet text to Devanagari only.
Rules:
- Output Devanagari script only for normal words.
- Do NOT translate; keep identical word order and meaning.
- Preserve punctuation, line breaks, and numerals exactly.
- Keep all-caps acronyms/tickers (A–Z) of length 2–6 in Latin unchanged unless they contain lowercase.
- Keep URLs, emails, and handles unchanged.
- Return ONLY the transformed text with no commentary.
""".strip()


def transliterate_to_devanagari(text: str) -> str:
    if not text:
        return ""

    client = _build_client()
    model_name = _get_model_name()
    user_prompt = (
        "Transliterate the following text to Devanagari following the rules exactly.\n"
        "Return ONLY the transformed text.\n\n"
        f"{text}"
    )

    result = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (result.choices[0].message.content or "").strip()
