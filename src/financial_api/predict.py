import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
)

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

def load_model():
    """Carga el modelo entrenado desde el archivo local."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def load_metadata() -> dict:
    """Carga los metadatos asociados al modelo entrenado."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos en: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def load_processed_data() -> pd.DataFrame:
    """Carga el dataset procesado utilizado para realizar predicciones."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado en: {PROCESSED_DATA_PATH}"
        )

    data = pd.read_csv(PROCESSED_DATA_PATH)

    if data.empty:
        raise ValueError("El dataset procesado está vacío.")

    return data


def get_latest_features(
    data: pd.DataFrame,
    symbol: str,
) -> tuple[pd.DataFrame, str]:
    """Obtiene las variables más recientes disponibles para un activo."""
    symbol = symbol.upper()

    symbol_data = data[
        data["symbol"] == symbol
    ].copy()

    if symbol_data.empty:
        available_symbols = sorted(
            data["symbol"].unique().tolist()
        )

        raise ValueError(
            f"Símbolo no disponible: {symbol}. "
            f"Símbolos disponibles: {available_symbols}"
        )

    symbol_data["date"] = pd.to_datetime(
        symbol_data["date"]
    )

    latest_row = (
        symbol_data
        .sort_values("date")
        .iloc[-1]
    )

    features = latest_row[
        FEATURE_COLUMNS
    ].to_frame().T

    prediction_date = (
        latest_row["date"]
        .date()
        .isoformat()
    )

    return features, prediction_date


def predict_symbol(symbol: str) -> dict:
    """Genera la predicción de tendencia para un activo financiero."""
    model = load_model()
    metadata = load_metadata()
    data = load_processed_data()

    features, prediction_date = get_latest_features(
        data,
        symbol,
    )

    prediction = int(
        model.predict(features)[0]
    )

    probability_up = float(
        model.predict_proba(features)[0][1]
    )

    prediction_label = (
        "up"
        if prediction == 1
        else "down"
    )

    return {
        "symbol": symbol.upper(),
        "prediction": prediction_label,
        "prediction_class": prediction,
        "probability_up": probability_up,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "prediction_horizon": metadata[
            "prediction_horizon_description"
        ],
        "data_date": prediction_date,
        "data_source": "cached",
        "disclaimer": metadata["disclaimer"],
    }


if __name__ == "__main__":
    result = predict_symbol("AAPL")

    print("\nPredicción generada correctamente:")

    for key, value in result.items():
        print(f"- {key}: {value}")