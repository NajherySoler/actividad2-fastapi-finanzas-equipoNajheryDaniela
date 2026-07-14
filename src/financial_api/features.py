from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_data.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"


def load_raw_data() -> pd.DataFrame:
    """Carga y valida el dataset financiero consolidado."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {RAW_DATA_PATH}"
        )

    data = pd.read_csv(RAW_DATA_PATH)

    if data.empty:
        raise ValueError("El dataset financiero está vacío.")

    return data


def prepare_base_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara las columnas básicas antes de crear las variables."""
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        by=["symbol", "date"]
    ).reset_index(drop=True)

    return data


def add_return_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula el retorno diario del precio de cierre para cada activo."""
    data = data.copy()

    data["daily_return"] = (
        data.groupby("symbol")["close"]
        .pct_change()
    )

    return data


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Crea variables con los retornos de los tres días anteriores."""
    data = data.copy()

    data["return_lag_1"] = (
        data.groupby("symbol")["daily_return"]
        .shift(1)
    )

    data["return_lag_2"] = (
        data.groupby("symbol")["daily_return"]
        .shift(2)
    )

    data["return_lag_3"] = (
        data.groupby("symbol")["daily_return"]
        .shift(3)
    )

    return data


def add_moving_average_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula medias móviles de 5 y 10 días para cada activo."""
    data = data.copy()

    data["moving_average_5"] = (
        data.groupby("symbol")["close"]
        .transform(lambda series: series.rolling(window=5).mean())
    )

    data["moving_average_10"] = (
        data.groupby("symbol")["close"]
        .transform(lambda series: series.rolling(window=10).mean())
    )

    return data


def add_volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula la volatilidad de 5 y 10 días para cada activo."""
    data = data.copy()

    data["volatility_5"] = (
        data.groupby("symbol")["daily_return"]
        .transform(lambda series: series.rolling(window=5).std())
    )

    data["volatility_10"] = (
        data.groupby("symbol")["daily_return"]
        .transform(lambda series: series.rolling(window=10).std())
    )

    return data


def add_price_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compara el precio de cierre con las medias móviles."""
    data = data.copy()

    data["close_to_ma_5"] = (
        data["close"] / data["moving_average_5"]
    )

    data["close_to_ma_10"] = (
        data["close"] / data["moving_average_10"]
    )

    return data


def add_volume_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula el cambio porcentual diario del volumen por activo."""
    data = data.copy()

    data["volume_change"] = (
        data.groupby("symbol")["volume"]
        .pct_change()
    )

    return data


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """Crea la variable objetivo de tendencia del día siguiente."""
    data = data.copy()

    data["next_close"] = (
        data.groupby("symbol")["close"]
        .shift(-1)
    )

    data["target"] = (
        data["next_close"] > data["close"]
    ).astype("Int64")

    data.loc[data["next_close"].isna(), "target"] = pd.NA

    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construye todas las variables necesarias para el modelo."""
    data = prepare_base_data(data)
    data = add_return_feature(data)
    data = add_lag_features(data)
    data = add_moving_average_features(data)
    data = add_volatility_features(data)
    data = add_price_ratio_features(data)
    data = add_volume_feature(data)
    data = add_target(data)

    return data


def clean_feature_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros incompletos y prepara el dataset final."""
    data = data.copy()

    feature_columns = [
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

    required_columns = feature_columns + ["target"]

    data = data.dropna(subset=required_columns)

    data["target"] = data["target"].astype(int)

    data = data.drop(columns=["next_close"])

    return data



def save_processed_data(data: pd.DataFrame) -> None:
    """Guarda el dataset final utilizado por el modelo."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(f"Dataset procesado guardado en: {PROCESSED_DATA_PATH}")


def run_feature_pipeline() -> pd.DataFrame:
    """Ejecuta el proceso completo de ingeniería de variables."""
    data = load_raw_data()
    data = build_features(data)
    data = clean_feature_dataset(data)
    save_processed_data(data)

    return data


if __name__ == "__main__":
    dataset = run_feature_pipeline()

    print("\nIngeniería de variables finalizada correctamente.")
    print(f"Total de registros procesados: {len(dataset)}")
    print(f"Total de columnas generadas: {len(dataset.columns)}")