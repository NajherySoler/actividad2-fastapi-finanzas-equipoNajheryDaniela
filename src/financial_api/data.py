from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def create_data_directory() -> None:
    """Crea el directorio de datos crudos si no existe."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_symbol_data(symbol: str) -> pd.DataFrame:
    """Descarga los datos históricos de un activo financiero."""
    print(f"Descargando datos de {symbol}...")

    data = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No se encontraron datos para el símbolo {symbol}.")

    return data


def clean_symbol_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Limpia y organiza los datos históricos de un activo financiero."""
    data = data.reset_index()

    data.columns = [
        column[0] if isinstance(column, tuple) else column
        for column in data.columns
    ]

    data.columns = [
    column.lower().replace(" ", "_")
    for column in data.columns
    ]

    data["symbol"] = symbol

    return data

def save_symbol_data(data: pd.DataFrame, symbol: str) -> None:
    """Guarda los datos históricos de un activo en un archivo CSV."""
    file_path = RAW_DATA_DIR / f"{symbol}.csv"
    data.to_csv(file_path, index=False)

    print(f"Datos de {symbol} guardados en: {file_path}")


def download_all_data() -> pd.DataFrame:
    """Descarga, limpia y guarda los datos de todos los activos."""
    create_data_directory()

    all_data = []

    for symbol in SYMBOLS:
        data = download_symbol_data(symbol)
        data = clean_symbol_data(data, symbol)
        save_symbol_data(data, symbol)

        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)

    combined_file_path = RAW_DATA_DIR / "market_data.csv"
    combined_data.to_csv(combined_file_path, index=False)

    print(f"Datos combinados guardados en: {combined_file_path}")

    return combined_data


if __name__ == "__main__":
    dataset = download_all_data()

    print("\nDescarga finalizada correctamente.")
    print(f"Total de registros descargados: {len(dataset)}")