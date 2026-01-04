import base64
import logging
import os
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

VOICEOVER_PROMPT = """
Create a YouTube video voiceover script for THIS image only, keep simple easy to understand english and hinglish.

STRICT RULES:
- Use ONLY the data and information visible on this image.
- DO NOT add any new data, assumptions, opinions, predictions, or external references.
- Generate ONE separate script for this slide only.
- Output ONLY the narration script with no labels, slide numbers, or metadata.

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
- Strictly no Special characters like "!@#$%^&*()_+"
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
    total_pages = len(images)

    for index, image in enumerate(images, start=1):
        logger.info("Calling OpenAI for page=%s model=%s", index, model)
        image_b64 = base64.b64encode(image).decode("utf-8")
        page_prompt = VOICEOVER_PROMPT
        if index == 1:
            page_prompt += (
                "\n\nFIRST SLIDE REQUIREMENTS:"  # noqa: E501
                "\n- Open with a warm greeting about pre market report.'"  # noqa: E501
                "\n- Immediately anchor the narration to the visible data on this slide keep it short and engaging, simple english no complex words."  # noqa: E501
            )
        if index == total_pages:
            page_prompt += (
                "\n\nFINAL SLIDE REQUIREMENTS:"  # noqa: E501
                "\n- Close with an engaging outro inviting viewer input, feedback, or questions about the day's data."  # noqa: E501
                "\n- Encourage the audience to react to insights across all slides without adding new information."  # noqa: E501
            )
        result = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": page_prompt},
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
