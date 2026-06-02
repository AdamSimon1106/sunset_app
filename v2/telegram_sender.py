import os
import requests

from susnset_forcast_generator import generate_forecast_text

# ========== CONFIG ==========
BOT_TOKEN = str(os.getenv("TELEGRAM_BOT_TOKEN")).strip()
CHANNEL_ID = str(os.getenv("TELEGRAM_CHANNEL_ID")).strip()
TELEGRAM_TIMEOUT_SECONDS = 30


def send_telegram_text(message: str) -> dict:
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": CHANNEL_ID,
            "text": message,
        },
        timeout=TELEGRAM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API returned failure: {data}")

    return data


def send_forecast() -> None:
    message = generate_forecast_text()
    result = send_telegram_text(message)
    print("[telegram] forecast message sent successfully")
    print(result)


if __name__ == "__main__":
    send_forecast()
