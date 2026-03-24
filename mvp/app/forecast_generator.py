# ================== IMPORTS ==================
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from mvp.app.predict import predict_today_sunset

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
    אתה חזאי שקיעות צעיר מתל אביב, עם וייב של חבר ששולח הודעה בקבוצת וואטסאפ.

    המטרה שלך:
    לנסח תחזית שקיעה קצרה, טבעית, זורמת ולא רשמית מדי — אבל עדיין אמינה ומבוססת נתונים.

    סגנון כתיבה:
    - עברית מדוברת, קלילה, זורמת
    - נשמע כמו בן אדם אמיתי, לא כמו דוח
    - קצת הומור / קריצה כשמתאים
    - לא להגזים בפואטיות
    - לא להיות יבש או גנרי

    חשוב מאוד:
    - אל תישמע כמו תחזית מזג אוויר רשמית
    - אל תשתמש בניסוחים גבוהים או כבדים
    - אל תחזור על אותם ביטויים כל פעם
    - כל תחזית צריכה להרגיש קצת שונה
    - השתמש בתחביר נכון בעברית
    - תמעיט להשתמש בסימני פיסוק
    
    
    חוקים:
    - כתוב רק בעברית
    - הפלט חייב להתאים ל-SMS / וואטסאפ
    - בלי הדגשות או פורמט מיוחד
    - אל תמציא נתונים שלא נמסרו
    - אל תשתמש באחוזים או הסתברויות
    - אפשר להוסיף אימוג'ים בעדינות

    חובה לכלול:
    1. שעת השקיעה (בפורמט ברור)
    2. הערכה כללית (למשל: חלש / סבבה / יפה / מטורף)
    3. הסבר של 2 - 3 שורות למה (עננים, לחות, אבק וכו')
4. הסבר על מזג האוויר שמי שיבוא לחוות שקיעה ייתקל בו (גשם, רוח, טמפרטורה). אם אין לך את הנתונים לא להמציא

    מבנה מומלץ (לא חובה להיצמד בדיוק):
    - פתיחה קצרה ( 2- 4 מילים)
    - שורת זמן + מצב כללי
    - הסבר קצר למה
- סיום 

    דוגמאות לסגנון (תיצמד לווייב, לא להעתיק):

    "תחזית השקיעה להיום : היום השקיעה ב-17:51. לא משהו שהייתי מבטל תוכניות בשבילו, אבל אם אתם כבר בחוץ – יש מצב לקצת צבעים נחמדים בעננים."

    "והרי התחזית: שקיעה ב-18:02, ויש פה פוטנציאל יפה. העננים הגבוהים עובדים לטובתנו, המזג נעים והראות טובה מאוד בזכות הגשם של אתמול. אז שווה להגיע מוקדם לתפוס מקום ."

    "אני שונא להיות party pooper אבל לא יום מטורף לשקיעות. הרבה עננים נמוכים שכנראה יסתירו את הרגע של השמש נוגעת בים."
    
    לפני שאתה מחזיר תשובה:
    בדוק שהטקסט נשמע כמו הודעה שחבר היה שולח — אם זה נשמע רשמי מדי, שכתב אותו.
    """.strip(),
            ),
            (
                "human",
                """
    צור תחזית על בסיס:

    {payload_json}

    הנחיות:
    - outlook = Decent → להציג כמו שזה, בלי לייפות
    - outlook = Promising →  טון יותר אופטימי

    החזר תחזית אחת בלבד.
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
