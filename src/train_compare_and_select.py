# ================== IMPORTS ==================
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight
from mvp.app.model_preprocessing import WeatherPreprocessor

# ================== CONFIG ===================
PATH_TO_RAW_DATA = "../data/raw/raw_sunsets_ranked.xlsx"
PATH_TO_FINAL_BUNDLE = "../mvp/models/sunset_best_pipeline.pkl"
PATH_TO_REPORT_JSON = "../models/sunset_model_comparison_report.json"

EXPORT_CLEAN_DATA = False
PATH_TO_CLEAN_DATA = "../data/clean/raw_sunsets_ranked_model_compare_clean.xlsx"

TARGET_COL = "beauty_score"
CLASS_TARGET_COL = "beauty_class"

RAW_FEATURE_COLS = [
    "cloud_cover_total",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "aerosol_optical_depth",
    "temperature",
    "dew_point",
    "pressure_6h_before",
    "pressure_trend",
    "relative_humidity",
    "pm25",
    "pm10",
    "wind_speed",
    "wind_direction",
]

BAD_MIN = 1
BAD_MAX = 3
OK_MIN = 4
OK_MAX = 6
GREAT_MIN = 7
GREAT_MAX = 10


# ================== HELPERS ==================
def ensure_output_dirs() -> None:
    Path(PATH_TO_FINAL_BUNDLE).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_TO_REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)
    if EXPORT_CLEAN_DATA:
        Path(PATH_TO_CLEAN_DATA).parent.mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def score_to_class(score: float) -> int:
    """
    1 = bad    (1..3)
    2 = ok     (4..6)
    3 = great  (7..10)
    """
    if score >= GREAT_MIN:
        return 3
    if score >= OK_MIN:
        return 2
    return 1


def class_to_label(class_id: int) -> str:
    return {
        1: "bad",
        2: "ok",
        3: "great",
    }[class_id]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = RAW_FEATURE_COLS + [TARGET_COL]
    validate_columns(df, required)

    df = df[df[TARGET_COL] != -1].copy()
    df = df[required].copy()

    # Optional: keep only realistic score range
    df = df[df[TARGET_COL].between(1, 10)].copy()

    df[CLASS_TARGET_COL] = df[TARGET_COL].apply(score_to_class)

    if EXPORT_CLEAN_DATA:
        df.to_excel(PATH_TO_CLEAN_DATA, index=False)

    return df

# ================== MODEL BUILDERS ==================
def build_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", WeatherPreprocessor(RAW_FEATURE_COLS, add_missing_indicators=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.03,
                    max_iter=400,
                    max_depth=6,
                    min_samples_leaf=10,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_classifier_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", WeatherPreprocessor(RAW_FEATURE_COLS, add_missing_indicators=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=400,
                    max_depth=6,
                    min_samples_leaf=10,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


# ================== EVALUATION ==================
def evaluate_regressor(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train_score: pd.Series,
    X_test: pd.DataFrame,
    y_test_score: pd.Series,
    y_test_class: pd.Series,
) -> dict:
    # CV on true regression target
    cv_mae = -cross_val_score(
        pipe,
        X_train,
        y_train_score,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_r2 = cross_val_score(
        pipe,
        X_train,
        y_train_score,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    pipe.fit(X_train, y_train_score)

    pred_score = pipe.predict(X_test)
    pred_score = np.clip(pred_score, 1, 10)
    pred_class = pd.Series(pred_score).apply(score_to_class).to_numpy()

    mae = mean_absolute_error(y_test_score, pred_score)
    r2 = r2_score(y_test_score, pred_score)

    acc = accuracy_score(y_test_class, pred_class)
    f1_macro = f1_score(y_test_class, pred_class, average="macro")
    f1_weighted = f1_score(y_test_class, pred_class, average="weighted")

    report = classification_report(
        y_test_class,
        pred_class,
        labels=[1, 2, 3],
        target_names=["bad", "ok", "great"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_class, pred_class, labels=[1, 2, 3])

    return {
        "model_family": "regression",
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "test_mae": float(mae),
        "test_r2": float(r2),
        "bucketed_accuracy": float(acc),
        "bucketed_f1_macro": float(f1_macro),
        "bucketed_f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_classifier(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train_class: pd.Series,
    X_test: pd.DataFrame,
    y_test_class: pd.Series,
) -> dict:
    cv_acc = cross_val_score(
        pipe,
        X_train,
        y_train_class,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_f1 = cross_val_score(
        pipe,
        X_train,
        y_train_class,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )

    # Need class-balanced weights for the final fit
    prep = pipe.named_steps["prep"]
    prep.fit(X_train)
    X_train_prepped = prep.transform(X_train)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_class,
    )

    pipe.named_steps["model"].fit(X_train_prepped, y_train_class, sample_weight=sample_weights)

    X_test_prepped = prep.transform(X_test)
    pred_class = pipe.named_steps["model"].predict(X_test_prepped)

    acc = accuracy_score(y_test_class, pred_class)
    f1_macro = f1_score(y_test_class, pred_class, average="macro")
    f1_weighted = f1_score(y_test_class, pred_class, average="weighted")

    report = classification_report(
        y_test_class,
        pred_class,
        labels=[1, 2, 3],
        target_names=["bad", "ok", "great"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_class, pred_class, labels=[1, 2, 3])

    # Rebuild fitted pipeline cleanly
    fitted_pipe = Pipeline(
        steps=[
            ("prep", prep),
            ("model", pipe.named_steps["model"]),
        ]
    )

    return {
        "model_family": "classification",
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "bucketed_accuracy": float(acc),
        "bucketed_f1_macro": float(f1_macro),
        "bucketed_f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "fitted_pipeline": fitted_pipe,
    }


def choose_best_model(reg_metrics: dict, clf_metrics: dict) -> str:
    reg_f1 = reg_metrics["bucketed_f1_macro"]
    clf_f1 = clf_metrics["bucketed_f1_macro"]

    # If one clearly wins, use it.
    if reg_f1 > clf_f1 + 0.01:
        return "regression"
    if clf_f1 > reg_f1 + 0.01:
        return "classification"

    # Tie-break: prefer regression because it stays closer to the real score.
    return "regression"


# ================== MAIN ===================
def main() -> None:
    ensure_output_dirs()

    df = load_data(PATH_TO_RAW_DATA)
    df = clean_data(df)

    X = df[RAW_FEATURE_COLS].copy()
    y_score = df[TARGET_COL].copy()
    y_class = df[CLASS_TARGET_COL].copy()

    X_train, X_test, y_train_score, y_test_score, y_train_class, y_test_class = train_test_split(
        X,
        y_score,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    reg_pipe = build_regression_pipeline()
    clf_pipe = build_classifier_pipeline()

    reg_metrics = evaluate_regressor(
        reg_pipe,
        X_train,
        y_train_score,
        X_test,
        y_test_score,
        y_test_class,
    )

    clf_metrics = evaluate_classifier(
        clf_pipe,
        X_train,
        y_train_class,
        X_test,
        y_test_class,
    )
    fitted_clf_pipe = clf_metrics.pop("fitted_pipeline")

    # Fit regressor once more for saving
    reg_pipe.fit(X_train, y_train_score)

    best_model_type = choose_best_model(reg_metrics, clf_metrics)
    best_pipeline = reg_pipe if best_model_type == "regression" else fitted_clf_pipe

    final_bundle = {
        "best_model_type": best_model_type,
        "pipeline": best_pipeline,
        "raw_feature_columns": RAW_FEATURE_COLS,
        "target_column": TARGET_COL,
        "class_target_column": CLASS_TARGET_COL,
        "class_mapping": {
            "bad": 1,
            "ok": 2,
            "great": 3,
        },
        "class_thresholds": {
            "bad_min": 1,
            "bad_max": 3,
            "ok_min": 4,
            "ok_max": 6,
            "great_min": 7,
            "great_max": 10,
        },
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
    }

    joblib.dump(final_bundle, PATH_TO_FINAL_BUNDLE)

    with open(PATH_TO_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_type": best_model_type,
                "regression_metrics": reg_metrics,
                "classification_metrics": clf_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== Regression model ===")
    print(f"CV MAE mean:        {reg_metrics['cv_mae_mean']:.4f} ± {reg_metrics['cv_mae_std']:.4f}")
    print(f"CV R² mean:         {reg_metrics['cv_r2_mean']:.4f} ± {reg_metrics['cv_r2_std']:.4f}")
    print(f"Test MAE:           {reg_metrics['test_mae']:.4f}")
    print(f"Test R²:            {reg_metrics['test_r2']:.4f}")
    print(f"Bucketed accuracy:  {reg_metrics['bucketed_accuracy']:.4f}")
    print(f"Bucketed F1 macro:  {reg_metrics['bucketed_f1_macro']:.4f}")
    print()

    print("=== Classification model ===")
    print(f"CV accuracy mean:   {clf_metrics['cv_accuracy_mean']:.4f} ± {clf_metrics['cv_accuracy_std']:.4f}")
    print(f"CV F1 macro mean:   {clf_metrics['cv_f1_macro_mean']:.4f} ± {clf_metrics['cv_f1_macro_std']:.4f}")
    print(f"Bucketed accuracy:  {clf_metrics['bucketed_accuracy']:.4f}")
    print(f"Bucketed F1 macro:  {clf_metrics['bucketed_f1_macro']:.4f}")
    print()

    print(f"=== Selected best model: {best_model_type} ===")
    print(f"Saved bundle to: {PATH_TO_FINAL_BUNDLE}")
    print(f"Saved report to: {PATH_TO_REPORT_JSON}")


if __name__ == "__main__":
    main()