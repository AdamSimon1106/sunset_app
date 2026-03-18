# ================== IMPORTS ==================
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mvp.app.forecast import build_today_model_input
from mvp.app.model_preprocessing import WeatherPreprocessor


# ================== CONFIG ===================
PATH_TO_MODEL = Path(__file__).resolve().parent.parent / "models" / "sunset_best_pipeline.pkl"


# ================== LOAD MODEL BUNDLE ===================
def load_bundle(path: Path = PATH_TO_MODEL) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("Model file must contain a dict bundle.")

    required_keys = {"best_model_type", "pipeline", "raw_feature_columns"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Model bundle missing keys: {sorted(missing_keys)}")

    return bundle


# ================== HELPERS ===================
def class_to_label(class_id: int) -> str:
    mapping = {
        1: "bad",
        2: "ok",
        3: "great",
    }
    if class_id not in mapping:
        raise ValueError(f"Unknown class id: {class_id}")
    return mapping[class_id]


def label_class(class_label: str) -> str:
    if class_label == "bad":
        return "Weak"
    if class_label == "ok":
        return "Decent"
    if class_label == "great":
        return "Promising"
    return "Unknown"


# ================== PREDICTION ===================
def predict_today_sunset(
    city_name: str,
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    target_date: str | None = None,
) -> dict[str, Any]:
    bundle = load_bundle()

    model_type = bundle["best_model_type"]
    pipeline = bundle["pipeline"]
    raw_feature_columns = bundle["raw_feature_columns"]

    forecast_result = build_today_model_input(
        city_name=city_name,
        latitude=latitude,
        longitude=longitude,
        feature_columns=raw_feature_columns,
        timezone=timezone,
        target_date=target_date,
        fail_on_missing=False,
    )

    sunset_time_today = forecast_result["sunset_time_today"]
    snapshot = forecast_result["snapshot"]
    missing_fields = forecast_result["missing_fields"]
    X_ready = forecast_result["X_ready"]

    prediction = pipeline.predict(X_ready)

    if len(prediction) != 1:
        raise ValueError(f"Expected exactly 1 prediction, got {len(prediction)}")

    result: dict[str, Any] = {
        "model_type": model_type,
        "city_name": city_name,
        "sunset_time_today": sunset_time_today,
        "snapshot": snapshot,
        "missing_fields_from_forecast": missing_fields,
        "X_ready": X_ready,
    }

    if model_type == "classification":
        predicted_class = int(prediction[0])
        predicted_label = class_to_label(predicted_class)

        result.update(
            {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "outlook": label_class(predicted_label),
            }
        )

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_ready)[0]
            result["class_probabilities"] = {
                "bad": float(probabilities[0]),
                "ok": float(probabilities[1]),
                "great": float(probabilities[2]),
            }

    elif model_type == "regression":
        score = float(prediction[0])

        if score < 4:
            predicted_class = 1
        elif score < 7:
            predicted_class = 2
        else:
            predicted_class = 3

        predicted_label = class_to_label(predicted_class)

        result.update(
            {
                "score": score,
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "outlook": label_class(predicted_label),
            }
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return result


# ================== FORMATTING ===================
def format_prediction_summary(result: dict[str, Any]) -> str:
    city_name = result["city_name"]
    sunset_time = result["sunset_time_today"]
    model_type = result["model_type"]
    predicted_label = result["predicted_label"]
    outlook = result["outlook"]

    lines = [
        f"Sunset prediction for {city_name}",
        f"Sunset time: {sunset_time}",
        f"Predicted class: {predicted_label}",
        f"Outlook: {outlook}",
        f"Model type: {model_type}",
    ]

    if "score" in result:
        lines.insert(2, f"Predicted beauty score: {result['score']:.2f}/10")

    if "class_probabilities" in result:
        probs = result["class_probabilities"]
        lines.append(
            "Class probabilities: "
            f"bad={probs['bad']:.3f}, "
            f"ok={probs['ok']:.3f}, "
            f"great={probs['great']:.3f}"
        )

    return "\n".join(lines)


# ================== MAIN ===================
if __name__ == "__main__":
    result = predict_today_sunset(
        city_name="Tel Aviv",
        latitude=32.0853,
        longitude=34.7818,
        timezone="auto",
        target_date=None,
    )

    print(format_prediction_summary(result))
    print("\nMissing fields reported by forecast.py:")
    print(result["missing_fields_from_forecast"])
    print("\nPrepared model input:")
    print(result["X_ready"].to_string(index=False))