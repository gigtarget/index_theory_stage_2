import logging
import os

from app.script_generator import _build_client
from app.youtube_uploader import build_description


logger = logging.getLogger(__name__)

DEFAULT_HEADER = (
    "Index Theory – Post Market Report (India)\n\n"
    "A structured post-market breakdown for Indian traders: price action, participation, "
    "risk signals, and key levels.\n"
)
DEFAULT_FOOTER = (
    "Disclaimer: Educational content only. Not investment advice.\n\n"
    "Follow Index Theory for daily post-market reports and data-driven market insights.\n"
)
DEFAULT_MAX_CHARS = 4500
MAX_SCRIPT_CHARS = 12000


def _get_header_footer() -> tuple[str, str]:
    header = os.getenv("YT_DESC_HEADER", "")
    footer = os.getenv("YT_DESC_FOOTER", "")
    return header, footer


def _get_max_chars() -> int:
    raw = os.getenv("YT_DESC_MAX_CHARS", str(DEFAULT_MAX_CHARS))
    try:
        limit = int(raw)
    except ValueError:
        logger.warning("Invalid YT_DESC_MAX_CHARS=%s; using %s", raw, DEFAULT_MAX_CHARS)
        return DEFAULT_MAX_CHARS
    return limit if limit > 0 else DEFAULT_MAX_CHARS


def _get_model_name() -> str:
    return (
        os.getenv("YT_DESC_MODEL")
        or os.getenv("MODEL_NAME")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.2"
    )


def trim_to_limit(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def build_description_static(date_str: str) -> str:
    header, footer = _get_header_footer()
    base = build_description(date_str)
    parts = []
    if header.strip():
        parts.append(header.strip())
    parts.append(base.strip())
    if footer.strip():
        parts.append(footer.strip())
    return "\n\n".join(parts).strip()


def _truncate_script(full_script_text: str) -> str:
    cleaned = full_script_text.strip()
    if len(cleaned) <= MAX_SCRIPT_CHARS:
        return cleaned
    return cleaned[-MAX_SCRIPT_CHARS:]


def _build_dynamic_middle(date_str: str, full_script_text: str) -> str:
    client = _build_client()
    model_name = _get_model_name()
    system_prompt = (
        "You are a strict extraction assistant for YouTube descriptions. "
        "Use ONLY facts explicitly stated in the script. Do NOT add or infer new data. "
        "If a section is not supported by the script, omit it entirely. "
        "Output should read like a concise report with multiple sections and bullets."
    )
    user_prompt = (
        "Build a report-style YouTube description middle section from the script.\n\n"
        f"Market Date: {date_str}\n\n"
        "Required format (headings + bullets only):\n"
        "Market Date: {date}\n"
        "Executive Summary\n"
        "- ...\n"
        "Index Performance\n"
        "- ...\n"
        "Market Tone\n"
        "- ...\n"
        "Institutional Activity\n"
        "- ...\n"
        "Key Technical Observations\n"
        "- ...\n\n"
        "Sector & Stock Highlights\n"
        "- ...\n"
        "Derivatives/Flows & Participation\n"
        "- ...\n"
        "Risk Notes & Volatility\n"
        "- ...\n"
        "What to Watch Next Session\n"
        "- ...\n\n"
        "Rules:\n"
        "- Always include Market Date line.\n"
        "- Include Executive Summary with 2-4 bullets, only if script supports it.\n"
        "- Include Index Performance only if script mentions Nifty/Bank Nifty/Sensex.\n"
        "- Include Market Tone only if script mentions sentiment/cautious/bull/bear/weak/strong.\n"
        "- Include Institutional Activity only if script mentions FII or DII.\n"
        "- Include Key Technical Observations only if script mentions pivot/support/resistance/RSI/EMA/VWAP.\n"
        "- Include Sector & Stock Highlights only if script mentions sectors or stocks.\n"
        "- Include Derivatives/Flows & Participation only if script mentions OI, PCR, options, futures, volume, or breadth.\n"
        "- Include Risk Notes & Volatility only if script mentions VIX, volatility, risk, or caution.\n"
        "- Always include What to Watch Next Session with bullets derived from script.\n"
        "- Use ONLY facts present in the script; omit unknowns.\n"
        "- Prefer 14-24 total bullets across sections; keep each bullet concise.\n\n"
        "Script:\n"
        f"{_truncate_script(full_script_text)}"
    )
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (result.choices[0].message.content or "").strip()


def build_dynamic_description(date_str: str, full_script_text: str) -> str:
    header = os.getenv("YT_DESC_HEADER", "").strip() or DEFAULT_HEADER.strip()
    footer = os.getenv("YT_DESC_FOOTER", "").strip() or DEFAULT_FOOTER.strip()
    try:
        middle = _build_dynamic_middle(date_str, full_script_text)
        if not middle:
            raise RuntimeError("Dynamic description returned empty output.")
        combined = "\n\n".join([header, middle, footer]).strip()
        return trim_to_limit(combined, _get_max_chars())
    except Exception as exc:
        logger.warning("Dynamic description failed; falling back to static. error=%s", exc)
        return trim_to_limit(build_description_static(date_str), _get_max_chars())
