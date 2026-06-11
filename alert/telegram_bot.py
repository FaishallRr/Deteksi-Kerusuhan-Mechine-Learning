import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self._available = bot_token and bot_token != "YOUR_TELEGRAM_BOT_TOKEN"

    async def _ensure_bot(self):
        if self.bot is None and self._available:
            try:
                from telegram import Bot
                self.bot = Bot(token=self.bot_token)
            except Exception as e:
                logger.warning(f"Telegram bot init failed: {e}")
                self._available = False

    async def send_alert(
        self,
        message: str,
        photo_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        if not self._available:
            logger.info(f"[MOCK TELEGRAM] Alert would be sent:\n{message[:100]}...")
            return

        await self._ensure_bot()
        if self.bot is None:
            return

        try:
            if photo_path and Path(photo_path).exists():
                with open(photo_path, "rb") as f:
                    await self.bot.send_photo(chat_id=self.chat_id, photo=f, caption=message)
            elif video_path and Path(video_path).exists():
                with open(video_path, "rb") as f:
                    await self.bot.send_video(chat_id=self.chat_id, video=f, caption=message)
            else:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info("Telegram alert sent successfully")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_alert_sync(self, message: str, photo_path: str = None, video_path: str = None):
        try:
            asyncio.run(self.send_alert(message, photo_path, video_path))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.send_alert(message, photo_path, video_path))
