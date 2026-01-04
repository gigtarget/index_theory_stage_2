import logging
import os
from typing import List

from google import genai

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


def _build_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=api_key)


def generate_scripts_from_images(images: List[bytes], model: str) -> List[str]:
    client = _build_client()
    scripts: List[str] = []

    for index, image in enumerate(images, start=1):
        logger.info("Calling Gemini for page=%s model=%s", index, model)
        result = client.models.generate_content(
            model=model,
            contents=[
                genai.types.Part.from_bytes(data=image, mime_type="image/png"),
                VOICEOVER_PROMPT,
            ],
        )
        logger.info("Received Gemini response for page=%s", index)
        scripts.append(result.text.strip())
    return scripts
