# ================== IMPORTS ==================
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from predict import predict_today_sunset


# ================== CONFIG ===================
DEFAULT_MODEL = "gemini-2.5-flash"


# ================== HELPERS ===================
def wind_direction_to_hebrew(degrees: Any) -> str:
    try:
        deg = float(degrees) % 360
    except (TypeError, ValueError):
        return "לא זמין"

    directions = [
        "צפונית",
        "צפון-מזרחית",
        "מזרחית",
        "דרום-מזרחית",
        "דרומית",
        "דרום-מערבית",
        "מערבית",
        "צפון-מערבית",
    ]
    index = round(deg / 45) % 8
    return directions[index]


def safe_float(value: Any, digits: int = 1) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_llm_weather_payload(prediction_result: dict[str, Any]) -> dict[str, Any]:
    snapshot = prediction_result.get("snapshot", {})

    wind_direction_deg = safe_float(snapshot.get("wind_direction"), 0)

    payload = {
        "city_name": prediction_result.get("city_name"),
        "sunset_time": prediction_result.get("sunset_time_today"),
        "predicted_label": prediction_result.get("predicted_label"),
        "outlook": prediction_result.get("outlook"),
        "model_type": prediction_result.get("model_type"),
        "cloud_cover_total": safe_float(snapshot.get("cloud_cover_total")),
        "cloud_cover_low": safe_float(snapshot.get("cloud_cover_low")),
        "cloud_cover_mid": safe_float(snapshot.get("cloud_cover_mid")),
        "cloud_cover_high": safe_float(snapshot.get("cloud_cover_high")),
        "aerosol_optical_depth": safe_float(snapshot.get("aerosol_optical_depth"), 2),
        "temperature": safe_float(snapshot.get("temperature")),
        "dew_point": safe_float(snapshot.get("dew_point")),
        "pressure_6h_before": safe_float(snapshot.get("pressure_6h_before")),
        "pressure_trend": safe_float(snapshot.get("pressure_trend")),
        "relative_humidity": safe_float(snapshot.get("relative_humidity")),
        "pm25": safe_float(snapshot.get("pm25")),
        "pm10": safe_float(snapshot.get("pm10")),
        "wind_speed": safe_float(snapshot.get("wind_speed")),
        "wind_direction": wind_direction_deg,
        "wind_direction_text": wind_direction_to_hebrew(wind_direction_deg),
    }

    if "score" in prediction_result:
        payload["score"] = safe_float(prediction_result.get("score"), 2)

    if "class_probabilities" in prediction_result:
        probs = prediction_result["class_probabilities"]
        payload["class_probabilities"] = {
            "bad": safe_float(probs.get("bad"), 3),
            "ok": safe_float(probs.get("ok"), 3),
            "great": safe_float(probs.get("great"), 3),
        }

    return payload


# ================== LLM GENERATION ===================
def generate_forecast_text(
    prediction_result: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> str:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

    payload = build_llm_weather_payload(prediction_result)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
אתה חזאי שקיעות צעיר, ישראלי, תל אביבי, הומוריסטי אבל מקצועי שכותב תחזיות קצרות, טבעיות וברורות בעברית.

המשימה שלך:
לקבל תוצאת חיזוי ונתוני מזג אוויר בזמן השקיעה, ולנסח תחזית שקיעה בעברית.

חוקים:
- כתוב רק בעברית.
-הפלט צריך להיות בטוח לשליחה ב SMS .
 - אל תדגיש טקסט. 
- אל תמציא נתונים שלא נמסרו.
- אל תשמש באחוזים או הסתברויות.
- אפשר להוסיף אימוג'ים בעדינות.
- הסתמך רק על המידע שקיבלת.
- חייבים להופיע בטקסט:
  1. שעת השקיעה
  2. תחזית כללית לאיכות השקיעה
  3. נימוק קצר לפי הנתונים
  4. תנאי הרוח בזמן השקיעה, כולל מהירות וכיוון אם קיימים
  5. האם יראו את השמש נוגעת באופק.
- שמור על ניסוח ידידותי וקצר.
                """.strip(),
            ),
            (
                "human",
                """
צור תחזית שקיעה בעברית על בסיס הנתונים הבאים:

{payload_json}

הנחיות פרשנות:
- קח בחשבון שבתל אביב, לצופה יש קו ראות ישר לאופק בים אז תנסה להבין איך יראו את השקיעה בסביבה כזו.
- השתמש בנתוני התחזית כדי להסביר בקצרה למה השקיעה צפויה להיות חלשה, סבירה או טובה או שיש לה פוטנציאל להיות מדהימה.
- ציין במפורש את שעת השקיעה.
- ציין במפורש את מצב הרוח בזמן השקיעה, ואת הטמפרטורה, תמיד תמליץ איך כדאי להתלבש.
- אם outlook הוא Decent, תנסח תחזית מאוזנת ולא מתלהבת מדי.
- אם outlook הוא Promising, אפשר לנסח תחזית יותר אופטימית.
- תרגיש חופשי לדבר בסלנג עדכני. השתמש במילים כמו "בדוק", "נדיר", "בסט", "בסדר פלוס" / "בסדר מינוס","רמה גבוהה", "על הפנים" וכו'
- שלב יציאות על דברים שאקטואלים לתל אביבים כמו אירועים בעיר, המלחמה או סתלבט כללי על תל אביב
- אם נראה שתהיה שקיעה פחות מרגשת תציג את המצב כמו שהוא, אל תנסה לשכנע אנשים לצאת לים
- אם נראה שתהיה שקיעה יפה מאוד אז תתנסח באופן שקצת "מוכר" את השקיעה אבל בטוב טעם
- אם outlook הוא Weak, תנסח בהתאם.
                """.strip(),
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "payload_json": json.dumps(payload, ensure_ascii=False, indent=2),
        }
    )

    text = response.content.strip()
    if not text:
        raise RuntimeError("Gemini returned an empty forecast")

    return text


# ================== MAIN ===================
if __name__ == "__main__":
    prediction_result = predict_today_sunset(
        city_name="Tel Aviv",
        latitude=32.0853,
        longitude=34.7818,
        timezone="auto",
        target_date=None,
    )

    forecast_text = generate_forecast_text(prediction_result)

    print("=== RAW PREDICTION ===")
    print(prediction_result)
    print("\n=== HEBREW FORECAST ===")
    print(forecast_text)