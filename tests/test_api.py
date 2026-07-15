from fastapi.testclient import TestClient
 
from financial_api.api import app
 
client = TestClient(app)


def test_health_returns_200():
    """Verifica que el endpoint de salud responda correctamente."""
    response = client.get("/health")
 
    assert response.status_code == 200


def test_health_response_structure():
    """Verifica la estructura de la respuesta del endpoint de salud."""
    response = client.get("/health")
    data = response.json()
 
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_model_metadata_returns_200():
    """Verifica que el endpoint de metadatos responda correctamente."""
    response = client.get("/model/metadata")
 
    assert response.status_code == 200

def test_model_metadata_response_structure():
    """Verifica la estructura básica de los metadatos del modelo."""
    response = client.get("/model/metadata")
    data = response.json()
 
    assert "model_name" in data
    assert "model_version" in data
    assert "feature_columns" in data
    assert "metrics" in data
    assert "disclaimer" in data

def test_market_data_returns_200():
    """Verifica que los datos de mercado de AAPL se obtengan correctamente."""
    response = client.get("/market-data/AAPL")
 
    assert response.status_code == 200

def test_market_data_response_structure():
    """Verifica la estructura de los datos de mercado."""
    response = client.get("/market-data/AAPL")
    data = response.json()
 
    assert data["symbol"] == "AAPL"
    assert "date" in data
    assert "close" in data
    assert "daily_return" in data
    assert "moving_average_5" in data
    assert "moving_average_10" in data
    assert "volatility_5" in data
    assert data["data_source"] == "cached"

def test_predict_returns_200():
    """Verifica que la predicción para AAPL responda correctamente."""
    response = client.post(
        "/predict",
        json={"symbol": "AAPL"},
    )
 
    assert response.status_code == 200

def test_predict_response_structure():
    """Verifica la estructura de la respuesta de predicción."""
    response = client.post(
        "/predict",
        json={"symbol": "AAPL"},
    )
 
    data = response.json()
 
    assert data["symbol"] == "AAPL"
    assert data["prediction"] in ["up", "down"]
    assert data["prediction_class"] in [0, 1]
    assert 0 <= data["probability_up"] <= 1
    assert "model_name" in data
    assert "model_version" in data
    assert "prediction_horizon" in data
    assert "data_date" in data
    assert data["data_source"] == "cached"
    assert "disclaimer" in data

def test_predict_rejects_unsupported_symbol():
    """Verifica que la API rechace símbolos no permitidos."""
    response = client.post(
        "/predict",
        json={"symbol": "TSLA"},
    )
 
    assert response.status_code == 422

def test_market_data_returns_404_for_unknown_symbol():
    """Verifica que un símbolo desconocido retorne 404."""
    response = client.get("/market-data/TSLA")
 
    assert response.status_code == 404

