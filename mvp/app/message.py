from mvp.app.forecast_generator import generate_forecast_text, predict_today_sunset
import os
import requests
#========== CONFIG ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "-1003771989105"


def send_message():
    prediction_result = predict_today_sunset(
        city_name="Tel Aviv",
        latitude=32.0853,
        longitude=34.7818,
        timezone="auto",
        target_date=None,
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    message = generate_forecast_text(prediction_result)
    response = requests.post(url, json={
        "chat_id": CHANNEL_ID,
        "text": message
    }, timeout=30)
    print(response.json())

if __name__ == "__main__":
     send_message()
