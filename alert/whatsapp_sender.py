import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WhatsAppAlert:
    def __init__(self, phone_number: str, api_key: str):
        self.phone_number = phone_number
        self.api_key = api_key
        self._available = bool(phone_number and phone_number != "YOUR_WHATSAPP_NUMBER")

    def send_alert(self, message: str, photo_path: Optional[str] = None):
        if not self._available:
            logger.info(f"[MOCK WHATSAPP] Alert would be sent to {self.phone_number}")
            return

        try:
            import pywhatkit as kit
            kit.sendwhatmsg_instantly(
                phone_no=self.phone_number,
                message=message,
                wait_time=10,
            )
            logger.info(f"WhatsApp alert sent to {self.phone_number}")
        except ImportError:
            logger.warning("pywhatkit not installed, skipping WhatsApp")
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
