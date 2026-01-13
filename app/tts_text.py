from openai import OpenAI


def hinglish_to_devanagari(
    text: str,
    *,
    model_name: str,
    client: OpenAI | None = None,
) -> str:
    active_client = client or OpenAI()
    system_prompt = (
        "You are a Hindi TTS assistant. Convert Hinglish narration into natural spoken Hindi "
        "in Devanagari script. Keep numbers and symbols exactly as provided. Return plain text only."
    )
    user_prompt = f"Rewrite this narration into spoken Hindi (Devanagari only):\n{text}"
    result = active_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_completion_tokens=2048,
    )
    msg = result.choices[0].message
    return (getattr(msg, "content", None) or "").strip()
