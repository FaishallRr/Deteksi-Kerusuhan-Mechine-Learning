from typing import Optional


class WhatsAppAlert:
    def __init__(self, phone_number: str, api_key: str):
        self.phone_number = phone_number
        self.api_key = api_key

    def send_alert(self, message: str, photo_path: Optional[str] = None):
        try:
            import pywhatkit as kit
            kit.sendwhatmsg_instantly(
                phone_no=self.phone_number,
                message=message,
                wait_time=10,
            )
        except Exception as e:
            print(f"WhatsApp send failed: {e}")
