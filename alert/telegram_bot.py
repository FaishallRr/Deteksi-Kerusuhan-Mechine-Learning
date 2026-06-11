import asyncio
from pathlib import Path
from typing import Optional


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None

    async def _ensure_bot(self):
        if self.bot is None:
            from telegram import Bot
            self.bot = Bot(token=self.bot_token)

    async def send_alert(
        self,
        message: str,
        photo_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        await self._ensure_bot()

        if photo_path and Path(photo_path).exists():
            with open(photo_path, "rb") as f:
                await self.bot.send_photo(chat_id=self.chat_id, photo=f, caption=message)
        elif video_path and Path(video_path).exists():
            with open(video_path, "rb") as f:
                await self.bot.send_video(chat_id=self.chat_id, video=f, caption=message)
        else:
            await self.bot.send_message(chat_id=self.chat_id, text=message)

    def send_alert_sync(self, message: str, photo_path: str = None, video_path: str = None):
        asyncio.run(self.send_alert(message, photo_path, video_path))
