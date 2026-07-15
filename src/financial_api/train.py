import json
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "daily_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "moving_average_5",
    "moving_average_10",
    "volatility_5",
    "volatility_10",
    "close_to_ma_5",
    "close_to_ma_10",
    "volume_change",
]


def load_processed_data() -> pd.DataFrame:
    """Carga y valida el dataset preparado para entrenamiento."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado en: {PROCESSED_DATA_PATH}"
        )

    data = pd.read_csv(PROCESSED_DATA_PATH)

    if data.empty:
        raise ValueError("El dataset procesado está vacío.")

    required_columns = FEATURE_COLUMNS + ["date", "symbol", "target"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {missing_columns}"
        )

    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa los datos en entrenamiento y prueba usando una fecha de corte."""
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        by=["date", "symbol"]
    ).reset_index(drop=True)

    unique_dates = sorted(data["date"].unique())

    split_index = int(len(unique_dates) * 0.80)
    split_date = unique_dates[split_index]

    train_data = data[data["date"] < split_date]
    test_data = data[data["date"] >= split_date]

    if train_data.empty or test_data.empty:
        raise ValueError(
            "No fue posible crear conjuntos de entrenamiento y prueba."
        )

    print(f"Fecha de corte utilizada: {pd.Timestamp(split_date).date()}")
    print(f"Registros de entrenamiento: {len(train_data)}")
    print(f"Registros de prueba: {len(test_data)}")

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["target"]

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["target"]

    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Entrena el modelo de clasificación Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    return model


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Entrena un modelo de clasificación Logistic Regression."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evalúa el modelo y devuelve sus métricas principales."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    return metrics


def save_metrics(metrics: dict) -> None:
    """Guarda las métricas de evaluación en un archivo JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Métricas guardadas en: {METRICS_PATH}")


def save_model(model: RandomForestClassifier) -> None:
    """Guarda el modelo entrenado para utilizarlo posteriormente en la API."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Modelo guardado en: {MODEL_PATH}")


def save_model_metadata(
    metrics: dict,
    data: pd.DataFrame,
    model_name: str,
) -> None:
    """Guarda la información descriptiva y técnica del modelo."""
    metadata = {
        "model_name": model_name,
        "model_version": f"{model_name.lower()}_v1",
        "task": "next_day_trend_classification",
        "training_date": datetime.now().isoformat(),
        "symbols": sorted(data["symbol"].unique().tolist()),
        "prediction_horizon": 1,
        "prediction_horizon_description": "next_day",
        "target": "next_day_price_increase",
        "feature_columns": FEATURE_COLUMNS,
        "main_metric": "f1_score",
        "metrics": metrics,
        "training_records": len(data),
        "disclaimer": (
            "Modelo desarrollado con fines educativos. "
            "No constituye asesoría financiera."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Metadatos guardados en: {METADATA_PATH}")


def run_training_pipeline() -> dict:
    """Entrena, compara y selecciona el mejor modelo."""
    data = load_processed_data()

    x_train, x_test, y_train, y_test = split_data(data)

    random_forest = train_model(
        x_train,
        y_train,
    )

    logistic_regression = train_logistic_regression(
        x_train,
        y_train,
    )

    random_forest_metrics = evaluate_model(
        random_forest,
        x_test,
        y_test,
    )

    logistic_regression_metrics = evaluate_model(
        logistic_regression,
        x_test,
        y_test,
    )

    print("\nMétricas de Random Forest:")
    for metric_name, metric_value in random_forest_metrics.items():
        print(f"- {metric_name}: {metric_value}")

    print("\nMétricas de Logistic Regression:")
    for metric_name, metric_value in logistic_regression_metrics.items():
        print(f"- {metric_name}: {metric_value}")

    if (
        logistic_regression_metrics["roc_auc"]
        > random_forest_metrics["roc_auc"]
    ):
        best_model = logistic_regression
        best_model_name = "LogisticRegression"
        best_metrics = logistic_regression_metrics
    else:
        best_model = random_forest
        best_model_name = "RandomForestClassifier"
        best_metrics = random_forest_metrics

    print(f"\nMejor modelo seleccionado: {best_model_name}")

    save_model(best_model)
    save_metrics(best_metrics)
    save_model_metadata(
    best_metrics,
    data,
    best_model_name,
    )

    return best_metrics

if __name__ == "__main__":
    metrics = run_training_pipeline()

    print("\nEntrenamiento finalizado correctamente.")
    print("Métricas obtenidas:")

    for metric_name, metric_value in metrics.items():
        print(f"- {metric_name}: {metric_value}")


