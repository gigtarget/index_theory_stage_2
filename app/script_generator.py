import base64
import logging
import os
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

VOICEOVER_PROMPT = """
You are a professional Indian stock market content writer and voiceover script specialist.

Mandatory Rules
- Use ONLY the data, numbers, statements, and insights present on the slide/page.
- DO NOT add new data, assumptions, opinions, predictions, or external references.
- Generate one separate script per slide. No combining.
- Language must be simple and clear, suitable for Indian retail traders.
- Tone: Professional, Serious, Calm, confident, authoritative, Slightly dramatic to maintain attention.
- The script should feel highly informative and valuable, even if the content is basic.
- Use confident framing.
- Emphasize clarity, structure, and relevance.
- Avoid hype or exaggeration.
- Use controlled Hinglish: Hindi words in proper हिंदी (Devanagari) only, English words in English only, No slang or casual YouTube language.
- Do NOT include emojis, jokes, overly emotional language, or personal opinions.
- Treat each slide independently. No summaries across slides.

Script Style Guidelines
- Start with context-setting lines for the slide.
- Explain what the slide means in practical terms.
- Maintain a subtle sense of urgency or importance.
- Address Indian retail traders directly.
- Use short, crisp sentences suitable for voiceover.

Output Format
- Write in paragraph form, optimized for voiceover.
- Do NOT mention instructions.
- Do NOT explain reasoning.
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
