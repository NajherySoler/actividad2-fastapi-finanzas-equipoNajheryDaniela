from typing import Literal
 
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Datos de entrada para solicitar una predicción."""
 
    symbol: Literal["AAPL", "MSFT", "GOOGL"] = Field(
        ...,
        description="Símbolo del activo financiero a predecir.",
        examples=["AAPL"],
    )

class PredictionResponse(BaseModel):

    """Respuesta generada por el modelo para un activo financiero."""
 
    symbol: str = Field(

        ...,

        description="Símbolo del activo financiero evaluado.",

        examples=["AAPL"],

    )
 
    prediction: Literal["up", "down"] = Field(

        ...,

        description="Tendencia estimada para el siguiente día.",

        examples=["down"],

    )
 
    prediction_class: int = Field(

        ...,

        description="Clase numérica de la predicción: 1 para subida y 0 para bajada.",

        examples=[0],

    )
 
    probability_up: float = Field(

        ...,

        ge=0,

        le=1,

        description="Probabilidad estimada de que el precio suba.",

        examples=[0.4069],

    )
 
    model_name: str = Field(

        ...,

        description="Nombre del modelo utilizado.",

        examples=["RandomForestClassifier"],

    )
 
    model_version: str = Field(

        ...,

        description="Versión del modelo utilizado.",

        examples=["randomforestclassifier_v1"],

    )
 
    prediction_horizon: str = Field(

        ...,

        description="Horizonte temporal de la predicción.",

        examples=["next_day"],

    )
 
    data_date: str = Field(

        ...,

        description="Fecha de los datos utilizados para la predicción.",

        examples=["2025-12-29"],

    )
 
    data_source: str = Field(

        ...,

        description="Origen de los datos utilizados.",

        examples=["cached"],

    )
 
    disclaimer: str = Field(

        ...,

        description="Advertencia sobre el uso educativo del modelo.",

    )
 
class ModelMetadataResponse(BaseModel):
    """Información técnica y descriptiva del modelo entrenado."""
 
    model_name: str
    model_version: str
    task: str
    training_date: str
    symbols: list[str]
    prediction_horizon: int
    prediction_horizon_description: str
    target: str
    feature_columns: list[str]
    main_metric: str
    metrics: dict
    training_records: int
    disclaimer: str

class HealthResponse(BaseModel):
    """Estado general de la API."""
 
    status: str = Field(
        ...,
        description="Estado actual del servicio.",
        examples=["ok"],
    )
 
    model_loaded: bool = Field(
        ...,
        description="Indica si el modelo entrenado está disponible.",
        examples=[True],
    )