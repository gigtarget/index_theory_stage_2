import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_BASE_URL = "https://api.telegram.org"


class TelegramClient:
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
        self.session = httpx.AsyncClient(base_url=f"{TELEGRAM_BASE_URL}/bot{self.bot_token}")

    async def close(self) -> None:
        await self.session.aclose()

    async def send_message(self, chat_id: int, text: str) -> None:
        logger.info("Sending Telegram message to chat_id=%s", chat_id)
        try:
            response = await self.session.post("/sendMessage", json={"chat_id": chat_id, "text": text})
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError:
            logger.exception("HTTP error while calling Telegram sendMessage")
            raise

        if not data.get("ok"):
            logger.error("Telegram sendMessage responded with error: %s", data)
            raise RuntimeError(f"Failed to send message: {data}")

        logger.info("Telegram sendMessage succeeded for chat_id=%s", chat_id)

    async def get_file(self, file_id: str) -> Dict[str, Any]:
        response = await self.session.get("/getFile", params={"file_id": file_id})
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Failed to fetch file: {data}")
        return data["result"]

    async def download_file(self, file_path: str) -> bytes:
        file_url = f"{TELEGRAM_BASE_URL}/file/bot{self.bot_token}/{file_path}"
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            return response.content
