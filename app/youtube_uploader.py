import base64
import binascii
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from httplib2 import HttpLib2Error

logger = logging.getLogger(__name__)

UPLOAD_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (HttpLib2Error, OSError, ConnectionError, TimeoutError)
DEFAULT_TAGS = [
    "index theory",
    "nifty 50",
    "bank nifty",
    "post market report",
    "fii",
    "dii",
]


def decode_b64_secrets_to_tmp() -> Optional[tuple[Path, Path]]:
    client_b64 = os.environ.get("YT_CLIENT_SECRETS_B64")
    token_b64 = os.environ.get("YT_TOKEN_B64")

    if not client_b64 or not token_b64:
        logger.warning(
            "YouTube upload skipped: missing YT_CLIENT_SECRETS_B64 or YT_TOKEN_B64."
        )
        return None
    logger.info(
        "YouTube secrets provided. client_len=%s token_len=%s",
        len(client_b64),
        len(token_b64),
    )

    tmp_dir = Path("/tmp/yt")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    client_path = tmp_dir / "client_secrets.json"
    token_path = tmp_dir / "token.json"

    try:
        client_path.write_bytes(base64.b64decode(client_b64))
        token_path.write_bytes(base64.b64decode(token_b64))
    except (ValueError, OSError, binascii.Error) as exc:
        logger.warning("Failed to decode YouTube secrets: %s", exc)
        return None

    logger.info(
        "Wrote YouTube secrets to %s and %s",
        client_path,
        token_path,
    )
    return client_path, token_path


def build_youtube_client() -> Optional[object]:
    decoded = decode_b64_secrets_to_tmp()
    if not decoded:
        return None

    _, token_path = decoded
    logger.info("Loading YouTube token from %s", token_path)
    try:
        credentials = Credentials.from_authorized_user_file(
            str(token_path), scopes=UPLOAD_SCOPES
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load YouTube token: %s", exc)
        return None

    logger.info(
        "YouTube credentials loaded. valid=%s expired=%s has_refresh=%s",
        credentials.valid,
        credentials.expired,
        bool(credentials.refresh_token),
    )
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to refresh YouTube credentials: %s", exc)
                return None
        else:
            logger.warning("YouTube credentials are invalid or missing refresh token.")
            return None

    logger.info("Building YouTube API client.")
    return build("youtube", "v3", credentials=credentials)


def build_description(date_str: str) -> str:
    template = os.environ.get("YT_DESCRIPTION_TEMPLATE")
    if template:
        return template.replace("{DATE}", date_str)
    return "\n".join(
        [
            "Post Market Report for {DATE}.",
            "",
            "Daily recap of Nifty 50, Bank Nifty, and key market movers.",
            "",
            "Disclaimer: This video is for educational purposes only.",
            "",
            "#nifty50 #banknifty #stockmarket",
        ]
    ).replace("{DATE}", date_str)


def _rfc3339_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def upload_video(
    file_path: str,
    title: str,
    description: str,
    tags: Iterable[str],
    category_id: str,
    publish_at_utc: datetime,
) -> str:
    service = build_youtube_client()
    if service is None:
        raise RuntimeError("YouTube client not configured")

    path = Path(file_path)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        logger.warning("Failed to stat video file %s: %s", path, exc)
        file_size = None

    tags_list = list(tags)
    logger.info(
        "Starting YouTube upload. path=%s size=%s title=%s tags=%s category=%s publish_at=%s",
        path,
        file_size,
        title,
        len(tags_list),
        category_id,
        _rfc3339_utc(publish_at_utc),
    )

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags_list,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": _rfc3339_utc(publish_at_utc),
        },
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    retry = 0
    max_retries = 9

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                logger.info("YouTube upload progress: %.1f%%", status.progress() * 100)
        except HttpError as err:
            if err.resp and err.resp.status in RETRYABLE_STATUS_CODES:
                retry = _sleep_backoff(retry, max_retries, err)
                continue
            raise
        except RETRYABLE_EXCEPTIONS as err:
            retry = _sleep_backoff(retry, max_retries, err)
            continue

    if "id" not in response:
        raise RuntimeError("Unexpected YouTube response without video id")
    return response["id"]


def _sleep_backoff(retry: int, max_retries: int, err: Exception) -> int:
    if retry >= max_retries:
        logger.error("YouTube upload failed after %s retries: %s", retry, err)
        raise err
    sleep_seconds = (2**retry) + random.random()
    logger.warning("Retrying YouTube upload in %.1fs after error: %s", sleep_seconds, err)
    time.sleep(sleep_seconds)
    return retry + 1
