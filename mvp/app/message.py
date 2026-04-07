from __future__ import annotations

import os
import requests

from mvp.app.forecast_generator import generate_forecast_text
from mvp.app.predict import (
    DataUnavailableError,
    ModelInputError,
    PredictionError,
    predict_today_sunset,
)

# ========== CONFIG ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "-1003771989105"
TELEGRAM_TIMEOUT_SECONDS = 30


def _send_telegram_text(message: str) -> dict:
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


def _build_fallback_message(error: Exception) -> str:
    if isinstance(error, DataUnavailableError):
        return (
            "תחזית השקיעה להיום לא יצאה הפעם.\n"
            "הייתה תקלה זמנית במשיכת נתוני מזג האוויר.\n"
            "ננסה שוב בהרצה הבאה."
        )

    if isinstance(error, ModelInputError):
        return (
            "תחזית השקיעה להיום לא נשלחה.\n"
            "חסרים נתונים קריטיים לקלט של המודל."
        )

    if isinstance(error, PredictionError):
        return (
            "תחזית השקיעה להיום לא נשלחה.\n"
            "הייתה תקלה פנימית בחישוב התחזית."
        )

    return (
        "תחזית השקיעה להיום לא נשלחה.\n"
        "אירעה תקלה לא צפויה."
    )


def send_message() -> None:
    try:
        prediction_result = predict_today_sunset(
            city_name="Tel Aviv",
            latitude=32.0853,
            longitude=34.7818,
            timezone="auto",
            target_date=None,
        )

        message = generate_forecast_text(prediction_result)
        telegram_result = _send_telegram_text(message)
        print("[message] forecast message sent successfully")
        print(telegram_result)

    except (DataUnavailableError, ModelInputError, PredictionError) as exc:
        print(f"[message] forecast pipeline failed: {exc}")
        fallback_message = _build_fallback_message(exc)

        try:
            telegram_result = _send_telegram_text(fallback_message)
            print("[message] fallback message sent successfully")
            print(telegram_result)
        except Exception as telegram_exc:
            print(f"[message] failed to send fallback telegram message: {telegram_exc}")
            raise

    except Exception as exc:
        print(f"[message] unexpected failure: {exc}")
        fallback_message = _build_fallback_message(exc)

        try:
            telegram_result = _send_telegram_text(fallback_message)
            print("[message] generic fallback message sent successfully")
            print(telegram_result)
        except Exception as telegram_exc:
            print(f"[message] failed to send generic fallback telegram message: {telegram_exc}")
            raise


if __name__ == "__main__":
    send_message()
