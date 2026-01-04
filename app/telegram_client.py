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

    async def set_webhook(self, url: str) -> Dict[str, Any]:
        logger.info("Setting Telegram webhook to %s", url)
        response = await self.session.post("/setWebhook", json={"url": url})
        logger.debug("Telegram setWebhook response: status=%s, body=%s", response.status_code, response.text)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            logger.error("Telegram setWebhook responded with error: %s", data)
            raise RuntimeError(f"Failed to set webhook: {data}")
        logger.info("Telegram webhook registered successfully")
        return data

    async def get_webhook_info(self) -> Dict[str, Any]:
        logger.info("Fetching Telegram webhook info")
        response = await self.session.get("/getWebhookInfo")
        logger.debug(
            "Telegram getWebhookInfo response: status=%s, body=%s", response.status_code, response.text
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            logger.error("Telegram getWebhookInfo responded with error: %s", data)
            raise RuntimeError(f"Failed to get webhook info: {data}")
        logger.info("Fetched Telegram webhook info")
        return data.get("result", {})

    async def send_message(self, chat_id: int, text: str) -> None:
        logger.info("Sending Telegram message to chat_id=%s", chat_id)
        try:
            response = await self.session.post("/sendMessage", json={"chat_id": chat_id, "text": text})
            logger.debug(
                "Telegram sendMessage response: status=%s, body=%s", response.status_code, response.text
            )
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
        logger.info("Requesting Telegram file metadata for file_id=%s", file_id)
        response = await self.session.get("/getFile", params={"file_id": file_id})
        logger.debug("Telegram getFile response: status=%s, body=%s", response.status_code, response.text)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            logger.error("Telegram getFile responded with error: %s", data)
            raise RuntimeError(f"Failed to fetch file: {data}")
        logger.info("Fetched Telegram file metadata for file_id=%s", file_id)
        return data["result"]

    async def download_file(self, file_path: str) -> bytes:
        file_url = f"{TELEGRAM_BASE_URL}/file/bot{self.bot_token}/{file_path}"
        logger.info("Downloading Telegram file from %s", file_url)
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            logger.debug("Telegram file download response: status=%s", response.status_code)
            response.raise_for_status()
            logger.info("Downloaded %s bytes from Telegram file", len(response.content))
            return response.content
