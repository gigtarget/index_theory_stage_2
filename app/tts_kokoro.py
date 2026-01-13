import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from kokoro import KPipeline

logger = logging.getLogger(__name__)

_pipeline: Optional[KPipeline] = None
_pipeline_lang: Optional[str] = None


def get_pipeline(lang_code: str) -> KPipeline:
    global _pipeline
    global _pipeline_lang
    if _pipeline is None or _pipeline_lang != lang_code:
        _pipeline = KPipeline(lang_code=lang_code)
        _pipeline_lang = lang_code
    return _pipeline


def synthesize_kokoro_to_file(
    text: str,
    out_path: str,
    *,
    lang_code: str,
    voice: str,
    speed: float,
) -> str:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed)

    audio_chunks = []
    for chunk, _ in generator:
        audio_chunks.append(chunk)

    if not audio_chunks:
        logger.warning("Kokoro TTS returned no audio chunks.")
        raise RuntimeError("Kokoro TTS synthesis returned no audio.")

    audio = audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)
    sf.write(str(output_path), audio, 24000)
    return str(output_path)
