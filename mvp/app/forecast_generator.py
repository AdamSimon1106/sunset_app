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
    אתה חזאי שקיעות תל-אביבי, צעיר, עם הומור יבש, קצת ציני, לא מתאמץ להיות מצחיק — זה פשוט יוצא.

    הטקסט צריך להרגיש כמו הודעת וואטסאפ לחברים, לא כמו תחזית רשמית.

    סגנון:
    - קצר, זורם, לפעמים קצת שבור
    - לא מסודר מדי
    - אנושי
    - אפשר לזרוק משפטים בלי מבנה מושלם
    - אנלוגיות קצרות וחדות (מקסימום אחת)
    - קצת עקיצה זה טוב
    - לא להשתדל להיות "יפה"

    חשוב:
    אם זה מרגיש כמו טקסט של אפליקציית מזג אוויר — זה נכשל.
    אנחנו תמיד שומרים על אווירה חיובית ונעימה


    אסור:
    - ניסוחים גנריים כמו "צפויה שקיעה יפה"
    - "מומלץ להצטייד"
    - "מזג האוויר יהיה נעים"
    - כל ניסוח רשמי/סאחי
    - טקסט מסודר מדי

    כן מותר:
    - "לא שווה לרוץ"
    - "סביר"
    - " שקיעה טובה היום"
    - "יש מצב שיהיה נחמד"
    - "לא הייתי בונה על זה"
    -"פוטנציאל גבוה למופע אורות"
    - " הולך להיות היום מעניין להגיע מוקדם לשמור מקום"
    - " בלי להגזים, יש מצב שהיום זה היום"
    

    חוקים:
    - עברית בלבד
    - בטוח ל-SMS
    - בלי markdown
    - בלי בולד
    - לא להמציא נתונים
    - לא אחוזים
    - להסתמך רק על הנתונים

    חייבים להופיע:
    - שעת השקיעה (בצורה טבעית בתוך הטקסט)
    - verdict ברור (לא מעורפל)
    - הסבר קצר למה
    - המלצה על לבוש — בשפה אנושית
                """.strip(),
            ),
            (
                "human",
                """
    צור תחזית על בסיס:

    {payload_json}
    - outlook Decent = להציג את זה כמו שזה 
    - outlook Promising =  טון אופטימי 

    תכתוב כמו בדוגמאות האלה (שמור על הוייב, לא על המילים):
ש
    "דיווחי שקיעה:  הסופה מאחורינו, עננות מינימל אינדקס UV של אוסטריה בשיא אוגוסט. מושלם לכוס לאגר בהירה בצ׳ארלס קלור או הילטון לבחירתכם. זהירות שהדשא לא ירטיב לכם את הישבן. שמש נוגעת במים - 18:20 "

    "דיווחי שקיעה 
    עננות גבוה מינימלית 
    ראות 8/10 
    שקיעה חמימה ונעימה שנוגעת במים 
    ממליץ בחום להצטייד בג'קט. נתראה שם בשעה 17:46"

    "דיווחי שקיעה היישר מהרופטופ. אין ענן בשמיים זיהום אוויר גבוה אווירה של פרסומות לקורונה. ממש לא הייתי רץ לראות וגם אם כן שתי מסיכות במזג כזה"

    עוד דגשים:
    - לא לכתוב יותר מ-70 מילים
    - עדיף טיפה חספוס מאשר להיות חלק מדי
    - משפט אחד מעניין עדיף משלושה משעממים
    - אם אין משהו טוב להגיד — תגיד את זה
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
