import base64
import logging
import os
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

VOICEOVER_PROMPT = """
Create a YouTube video voiceover script for THIS slide/page only.

STRICT RULES:
- Use ONLY the data and information visible on this slide/page.
- DO NOT add any new data, assumptions, opinions, predictions, or external references.
- Generate ONE separate script for this slide only.

STYLE & TONE:
- This is for a YouTube market analysis video.
- Audience: Indian retail stock market traders.
- Language: Mostly English, with a few relevant Hindi words for connection.
- Tone: Professional, serious, calm, confident, slightly dramatic to retain attention.
- Frame the content so it feels informative, structured, and valuable.
- No hype, no exaggeration.

LANGUAGE RULES:
- Hindi words MUST be written in proper हिंदी (Devanagari).
- English words MUST remain in English.
- No slang, jokes, emojis, or casual YouTube language.

FORMAT:
- Short, clear paragraphs suitable for voiceover narration.
- Do NOT mention instructions or explain reasoning.
""".strip()



def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def generate_scripts_from_images(images: List[bytes], model: str) -> List[str]:
    client = _build_client()
    scripts: List[str] = []

    for index, image in enumerate(images, start=1):
        logger.info("Calling OpenAI for page=%s model=%s", index, model)
        image_b64 = base64.b64encode(image).decode("utf-8")
        result = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VOICEOVER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        )
        logger.info("Received OpenAI response for page=%s", index)
        scripts.append(result.choices[0].message.content.strip())
    return scripts
