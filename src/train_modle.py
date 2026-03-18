# ================== IMPORTS ==================
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor


# ================== CONFIG ===================
PATH_TO_RAW_DATA = "../data/raw/raw_sunsets_ranked.xlsx"
PATH_TO_MODEL = "../models/sunset_beauty_model.pkl"
PATH_TO_FEATURES_JSON = "../models/feature_columns.json"

# Set to True only if you want to inspect the cleaned training table
EXPORT_CLEAN_DATA = False
PATH_TO_CLEAN_DATA = "../data/clean/raw_sunsets_ranked_clean.xlsx"

TARGET_COL = "beauty_score"

# Only include features you can reasonably fetch again from a weather API.
# These are all weather / air-quality style fields.
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

# Columns treated as continuous for median imputation
CONTINUOUS_COLS = RAW_FEATURE_COLS.copy()


# ================== HELPERS ==================
def ensure_output_dirs() -> None:
    Path(PATH_TO_MODEL).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_TO_FEATURES_JSON).parent.mkdir(parents=True, exist_ok=True)
    if EXPORT_CLEAN_DATA:
        Path(PATH_TO_CLEAN_DATA).parent.mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the raw features we want plus the target.
    Remove unusable labels (-1).
    """
    df = df.copy()

    required = RAW_FEATURE_COLS + [TARGET_COL]
    validate_columns(df, required)

    df = df[df[TARGET_COL] != -1].copy()
    df = df[required].copy()

    if EXPORT_CLEAN_DATA:
        df.to_excel(PATH_TO_CLEAN_DATA, index=False)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only engineer features that can also be reproduced later
    from weather API data.
    """
    df = df.copy()

    # ---------------------------
    # Interaction features
    # ---------------------------
    if {"pressure_trend", "cloud_cover_total"}.issubset(df.columns):
        df["pressure_cloud"] = df["pressure_trend"] * df["cloud_cover_total"]

    if {"wind_speed", "cloud_cover_total"}.issubset(df.columns):
        df["wind_cloud"] = df["wind_speed"] * df["cloud_cover_total"]

    if {"pm10", "cloud_cover_total"}.issubset(df.columns):
        df["pm10_cloud"] = df["pm10"] * df["cloud_cover_total"]

    if {"aerosol_optical_depth", "relative_humidity"}.issubset(df.columns):
        df["aod_humidity"] = df["aerosol_optical_depth"] * df["relative_humidity"]

    # ---------------------------
    # Nonlinear features
    # ---------------------------
    if "cloud_cover_total" in df.columns:
        df["cloud_cover_total_sq"] = df["cloud_cover_total"] ** 2

    if "cloud_cover_high" in df.columns:
        df["cloud_cover_high_sq"] = df["cloud_cover_high"] ** 2

    if "aerosol_optical_depth" in df.columns:
        df["aod_sq"] = df["aerosol_optical_depth"] ** 2

    # ---------------------------
    # Physically meaningful derived features
    # ---------------------------
    # Smaller spread can suggest more moisture / haze
    if {"temperature", "dew_point"}.issubset(df.columns):
        df["dew_temp_spread"] = df["temperature"] - df["dew_point"]

    # Total particulate load
    if {"pm25", "pm10"}.issubset(df.columns):
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)

    return df


def add_missing_indicators(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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


def build_model() -> HistGradientBoostingRegressor:
    """
    HistGradientBoosting often works better on numeric tabular data.
    """
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=400,
        max_depth=6,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Evaluate both baseline and trained model.
    """
    # Baseline: always predict train mean
    baseline_pred = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    # Cross-validation on training split
    cv_mae_scores = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_r2_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    # Fit on training set
    model.fit(X_train, y_train)

    # Test predictions
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
        "cv_mae_mean": cv_mae_scores.mean(),
        "cv_mae_std": cv_mae_scores.std(),
        "cv_r2_mean": cv_r2_scores.mean(),
        "cv_r2_std": cv_r2_scores.std(),
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_predictions": test_pred,
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
        "raw_feature_columns": raw_feature_columns,
        "feature_columns": feature_columns,
        "train_medians": train_medians,
        "missing_indicator_columns": missing_indicator_columns,
        "target_column": TARGET_COL,
        "engineered_features": [
            "pressure_cloud",
            "wind_cloud",
            "pm10_cloud",
            "aod_humidity",
            "cloud_cover_total_sq",
            "cloud_cover_high_sq",
            "aod_sq",
            "dew_temp_spread",
            "pm_ratio",
        ],
    }

    joblib.dump(bundle, PATH_TO_MODEL)

    with open(PATH_TO_FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)


# ================== MAIN ===================
def main() -> None:
    ensure_output_dirs()

    # 1. Load
    df = load_data(PATH_TO_RAW_DATA)

    # 2. Clean
    df = clean_data(df)

    # 3. Engineer features
    df = engineer_features(df)

    # 4. Split X / y
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 6. Missing indicators
    X_train, X_test, missing_indicator_columns = add_missing_indicators(X_train, X_test)

    # 7. Impute
    X_train, X_test, train_medians = impute_with_train_medians(X_train, X_test)

    # 8. Final feature order
    feature_columns = X_train.columns.tolist()
    X_test = X_test[feature_columns].copy()

    # 9. Build model
    model = build_model()

    # 10. Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 11. Print results
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print()

    print("=== Baseline ===")
    print(f"Baseline MAE: {metrics['baseline_mae']:.4f}")
    print(f"Baseline R²:  {metrics['baseline_r2']:.4f}")
    print()

    print("=== Cross-Validation (train only) ===")
    print(f"CV MAE mean: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
    print(f"CV R² mean:  {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print()

    print("=== Test Set ===")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²:  {metrics['test_r2']:.4f}")
    print()

    # 12. Save trained bundle
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