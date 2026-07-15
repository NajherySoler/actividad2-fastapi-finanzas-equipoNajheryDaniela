import pandas as pd
from fastapi import FastAPI, HTTPException

 
from financial_api.predict import (
    load_metadata,
    load_model,
    load_processed_data,
    predict_symbol,
)

from financial_api.schemas import (
    HealthResponse,
    MarketDataResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
 
app = FastAPI(
    title="Financial Prediction API",
    description=(
        "API para generar predicciones educativas sobre la tendencia "
        "del precio de activos financieros."
    ),
    versión="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """Verifica que la API y el modelo estén disponibles."""
    try:
        load_model()
        load_metadata()
 
        return HealthResponse(
            status="ok",
            model_loaded=True,
        )
 
    except FileNotFoundError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
        )
    

@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    tags=["Model"],
)
def get_model_metadata() -> ModelMetadataResponse:
    """Devuelve los metadatos del modelo entrenado."""
    try:
        metadata = load_metadata()
 
        return ModelMetadataResponse(**metadata)
 
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Genera una predicción para el activo financiero solicitado."""
    try:
        result = predict_symbol(request.symbol)
 
        return PredictionResponse(**result)
 
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
 
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    
@app.get(
    "/market-data/{symbol}",
    response_model=MarketDataResponse,
    tags=["Market Data"],
)
def get_market_data(symbol: str) -> MarketDataResponse:
    """Devuelve la información financiera más reciente de un activo."""
    try:
        data = load_processed_data()
 
        symbol = symbol.upper()
 
        symbol_data = data[
            data["symbol"] == symbol
        ].copy()
 
        if symbol_data.empty:
            raise ValueError(
                f"Símbolo no disponible: {symbol}"
            )
 
        symbol_data["date"] = pd.to_datetime(
            symbol_data["date"]
        )
 
        latest_row = (
            symbol_data
            .sort_values("date")
            .iloc[-1]
        )
 
        return MarketDataResponse(
            symbol=symbol,
            date=latest_row["date"].date().isoformat(),
            close=float(latest_row["close"]),
            daily_return=float(latest_row["daily_return"]),
            moving_average_5=float(latest_row["moving_average_5"]),
            moving_average_10=float(latest_row["moving_average_10"]),
            volatility_5=float(latest_row["volatility_5"]),
            data_source="cached",
        )
 
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
 
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error