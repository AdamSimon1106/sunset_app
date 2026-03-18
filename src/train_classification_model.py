# ================== IMPORTS ==================
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


# ================== CONFIG ===================
PATH_TO_RAW_DATA = "../data/raw/raw_sunsets_ranked.xlsx"
PATH_TO_MODEL = "../models/sunset_beauty_classifier.pkl"
PATH_TO_FEATURES_JSON = "../models/feature_columns_classifier.json"

EXPORT_CLEAN_DATA = False
PATH_TO_CLEAN_DATA = "../data/clean/raw_sunsets_ranked_classified_clean.xlsx"

TARGET_COL = "beauty_score"
CLASS_TARGET_COL = "beauty_class"

# Only raw features to train on
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


# ================== HELPERS ==================
def ensure_output_dirs() -> None:
    Path(PATH_TO_MODEL).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_TO_FEATURES_JSON).parent.mkdir(parents=True, exist_ok=True)
    if EXPORT_CLEAN_DATA:
        Path(PATH_TO_CLEAN_DATA).parent.mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def map_beauty_score_to_class(score: float) -> int:
    """
    1 = bad    (1 <= score <= 3)
    2 = ok     (4 <= score <= 6)
    3 = great  (7 <= score <= 10)
    """
    if score >= 7:
        return 3
    if score >= 4:
        return 2
    return 1


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the raw features we want plus the target.
    Remove unusable labels (-1).
    Convert beauty_score into beauty_class.
    """
    df = df.copy()

    required = RAW_FEATURE_COLS + [TARGET_COL]
    validate_columns(df, required)

    df = df[df[TARGET_COL] != -1].copy()
    df = df[required].copy()
    df[CLASS_TARGET_COL] = df[TARGET_COL].apply(map_beauty_score_to_class)

    if EXPORT_CLEAN_DATA:
        df.to_excel(PATH_TO_CLEAN_DATA, index=False)

    return df


def add_missing_indicators(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Add *_missing columns before imputation.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    original_cols = X_train.columns.tolist()
    missing_indicator_cols = []

    for col in original_cols:
        indicator_col = f"{col}_missing"
        X_train[indicator_col] = X_train[col].isna().astype(int)
        X_test[indicator_col] = X_test[col].isna().astype(int)
        missing_indicator_cols.append(indicator_col)

    return X_train, X_test, missing_indicator_cols


def impute_with_train_medians(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fill numeric NaNs using train medians only.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_medians = {}

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            median_value = X_train[col].median()
            train_medians[col] = median_value
            X_train[col] = X_train[col].fillna(median_value)
            X_test[col] = X_test[col].fillna(median_value)

    return X_train, X_test, train_medians


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=400,
        max_depth=6,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Evaluate baseline, cross-validation, and test performance.
    """
    baseline_class = y_train.mode().iloc[0]
    baseline_pred = [baseline_class] * len(y_test)

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_f1_macro = f1_score(y_test, baseline_pred, average="macro")

    cv_accuracy_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    cv_f1_macro_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, test_pred, average="weighted")

    report = classification_report(
        y_test,
        test_pred,
        labels=[1, 2, 3],
        target_names=["bad", "ok", "great"],
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, test_pred, labels=[1, 2, 3])

    return {
        "baseline_class": int(baseline_class),
        "baseline_accuracy": baseline_accuracy,
        "baseline_f1_macro": baseline_f1_macro,
        "cv_accuracy_mean": cv_accuracy_scores.mean(),
        "cv_accuracy_std": cv_accuracy_scores.std(),
        "cv_f1_macro_mean": cv_f1_macro_scores.mean(),
        "cv_f1_macro_std": cv_f1_macro_scores.std(),
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "test_predictions": test_pred,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def save_bundle(
    model,
    feature_columns: list[str],
    train_medians: dict,
    raw_feature_columns: list[str],
    missing_indicator_columns: list[str],
) -> None:
    bundle = {
        "model": model,
        "model_type": "classifier",
        "raw_feature_columns": raw_feature_columns,
        "feature_columns": feature_columns,
        "train_medians": train_medians,
        "missing_indicator_columns": missing_indicator_columns,
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
        "engineered_features": [],
    }

    joblib.dump(bundle, PATH_TO_MODEL)

    with open(PATH_TO_FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)


# ================== MAIN ===================
def main() -> None:
    ensure_output_dirs()

    # 1. Load
    df = load_data(PATH_TO_RAW_DATA)

    # 2. Clean + target mapping
    df = clean_data(df)

    # 3. Split X / y
    X = df[RAW_FEATURE_COLS].copy()
    y = df[CLASS_TARGET_COL].copy()

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5. Missing indicators
    X_train, X_test, missing_indicator_columns = add_missing_indicators(X_train, X_test)

    # 6. Impute
    X_train, X_test, train_medians = impute_with_train_medians(X_train, X_test)

    # 7. Final feature order
    feature_columns = X_train.columns.tolist()
    X_test = X_test[feature_columns].copy()

    # 8. Build model
    model = build_model()

    # 9. Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 10. Print results
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print()

    print("=== Class Distribution ===")
    print("Train:")
    print(y_train.value_counts().sort_index())
    print()
    print("Test:")
    print(y_test.value_counts().sort_index())
    print()

    print("=== Baseline ===")
    print(f"Baseline class:     {metrics['baseline_class']}")
    print(f"Baseline accuracy:  {metrics['baseline_accuracy']:.4f}")
    print(f"Baseline F1 macro:  {metrics['baseline_f1_macro']:.4f}")
    print()

    print("=== Cross-Validation (train only) ===")
    print(f"CV accuracy mean:   {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"CV F1 macro mean:   {metrics['cv_f1_macro_mean']:.4f} ± {metrics['cv_f1_macro_std']:.4f}")
    print()

    print("=== Test Set ===")
    print(f"Test accuracy:      {metrics['test_accuracy']:.4f}")
    print(f"Test F1 macro:      {metrics['test_f1_macro']:.4f}")
    print(f"Test F1 weighted:   {metrics['test_f1_weighted']:.4f}")
    print()

    print("=== Classification Report ===")
    print(metrics["classification_report"])
    print()

    print("=== Confusion Matrix ===")
    print("Rows = true, Cols = predicted")
    print("Order: [bad=1, ok=2, great=3]")
    print(metrics["confusion_matrix"])
    print()

    # 11. Save trained bundle
    save_bundle(
        model=model,
        feature_columns=feature_columns,
        train_medians=train_medians,
        raw_feature_columns=RAW_FEATURE_COLS,
        missing_indicator_columns=missing_indicator_columns,
    )

    print(f"Model saved to: {PATH_TO_MODEL}")
    print(f"Feature list saved to: {PATH_TO_FEATURES_JSON}")


if __name__ == "__main__":
    main()