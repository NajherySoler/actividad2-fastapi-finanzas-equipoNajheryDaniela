# API de Predicción de Tendencias Financieras con Machine Learning

Proyecto desarrollado como parte de la segunda entrega del curso de MLOps de la Maestría en Ciencia de Datos.

El proyecto implementa un flujo completo de Machine Learning para la predicción de la tendencia del precio de activos financieros. La solución integra la obtención y preparación de datos históricos, ingeniería de variables, entrenamiento y evaluación de modelos, generación de predicciones, exposición del modelo mediante una API REST con FastAPI, pruebas automatizadas y contenerización con Docker.

## Integrantes

- Najhery Soler
- Laura

## Objetivo del proyecto

Desarrollar una solución reproducible de Machine Learning capaz de estimar si el precio de cierre de un activo financiero subirá o bajará en el siguiente día, utilizando información histórica del mercado.

La solución permite:

- Obtener y consolidar datos históricos de activos financieros.
- Generar variables predictoras a partir de series temporales.
- Entrenar y comparar diferentes modelos de clasificación.
- Evaluar el desempeño mediante métricas de clasificación.
- Seleccionar y serializar el mejor modelo.
- Generar predicciones para activos soportados.
- Exponer las funcionalidades mediante una API REST.
- Validar la API mediante pruebas automatizadas.
- Ejecutar la solución dentro de un contenedor Docker.

## Activos soportados

Actualmente, el proyecto trabaja con los siguientes símbolos:

- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation.
- `GOOGL` - Alphabet Inc.

## Tecnologías utilizadas

- Python 3.12
- Poetry
- pandas
- NumPy
- scikit-learn
- yfinance
- FastAPI
- Pydantic
- Uvicorn
- pytest
- Docker
- Git y GitHub

## Arquitectura de la solución

La solución se encuentra organizada en nueve etapas principales:

| Etapa | Descripción |
|---|---|
| Ingesta de datos | Obtención y consolidación de información histórica de los activos financieros. |
| Ingeniería de variables | Transformación de los datos y generación de características predictoras. |
| Preparación del dataset | Construcción del conjunto de datos utilizado para el entrenamiento. |
| Entrenamiento | Entrenamiento de diferentes modelos de clasificación. |
| Evaluación | Comparación de los modelos mediante métricas de desempeño. |
| Serialización | Almacenamiento del mejor modelo, sus métricas y metadatos. |
| Predicción | Generación de predicciones para los activos soportados. |
| API REST | Exposición del modelo y los datos mediante FastAPI. |
| Pruebas y despliegue | Validación automatizada y ejecución de la solución mediante Docker. |

### Flujo de procesamiento

**Datos históricos → Ingeniería de variables → Dataset procesado → Entrenamiento y evaluación → Selección del modelo → Predicción → API REST → Docker**

## Organización del proyecto

El repositorio se encuentra organizado por componentes funcionales:

### Datos

- `data/raw/market_data.csv`: contiene los datos históricos consolidados.
- `data/processed/model_dataset.csv`: contiene el dataset final utilizado por los modelos.

### Código fuente

El código principal se encuentra en `src/financial_api/`:

- `data.py`: obtención y consolidación de los datos históricos.
- `features.py`: procesamiento e ingeniería de variables.
- `train.py`: entrenamiento, evaluación y selección del modelo.
- `predict.py`: carga del modelo y generación de predicciones.
- `schemas.py`: esquemas de entrada y salida definidos con Pydantic.
- `api.py`: implementación de los endpoints mediante FastAPI.

### Artefactos y resultados

- `artifacts/model.joblib`: modelo entrenado y serializado.
- `artifacts/model_metadata.json`: información y metadatos del modelo.
- `reports/metrics.json`: métricas obtenidas durante la evaluación.

### Pruebas

- `tests/test_api.py`: pruebas automatizadas para validar los principales endpoints y escenarios de error de la API.

### Contenerización

- `Dockerfile`: configuración utilizada para construir la imagen de la aplicación.
- `.dockerignore`: archivos y directorios excluidos del contexto de construcción de Docker.

### Gestión del proyecto

- `pyproject.toml`: configuración del proyecto y sus dependencias.
- `poetry.lock`: versiones exactas de las dependencias instaladas.
- `README.md`: documentación general del proyecto.

## Instalación y configuración

### Requisitos previos

Para ejecutar el proyecto localmente se requiere:

- Python `>=3.12,<3.13`
- Poetry
- Git
- Docker, para la ejecución contenerizada

### Clonar el repositorio

```bash
git clone https://github.com/NajherySoler/actividad2-fastapi-finanzas-equipoNajheryDaniela.git
cd actividad2-fastapi-finanzas-equipoNajheryDaniela
```

### Instalar las dependencias

El proyecto utiliza Poetry para la gestión de dependencias y del entorno virtual:

```bash
poetry install
```

Para verificar la versión de Python utilizada:

```bash
poetry run python --version
```

## Ejecución del flujo de Machine Learning

### 1. Obtener los datos

```bash
poetry run python -m financial_api.data
```

Este proceso obtiene y consolida los datos históricos de los activos financieros soportados.

### 2. Generar las variables del modelo

```bash
poetry run python -m financial_api.features
```

El proceso genera el dataset procesado utilizado para el entrenamiento del modelo.

### 3. Entrenar y evaluar los modelos

```bash
poetry run python -m financial_api.train
```

Durante el entrenamiento se:

- realiza una división temporal entre entrenamiento y prueba;
- entrenan los modelos Random Forest y Logistic Regression;
- calculan las métricas de evaluación;
- selecciona el modelo con mejor desempeño según ROC-AUC;
- guarda el modelo seleccionado;
- generan las métricas y los metadatos del modelo.

### 4. Generar una predicción desde Python

```bash
poetry run python -m financial_api.predict
```

## Ejecución de la API

Para iniciar la API localmente:

```bash
poetry run uvicorn financial_api.api:app --reload
```

La aplicación estará disponible en:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`


## Endpoints de la API
 
La aplicación expone una API REST desarrollada con FastAPI para consultar el estado del servicio, acceder a información del modelo, consultar datos financieros y generar predicciones.
 
### Estado del servicio
 
`GET /health`
 
Permite verificar que la API esté disponible y que los artefactos del modelo puedan cargarse correctamente.
 
Ejemplo de respuesta:
 
```json

{

  "status": "ok",

  "model_loaded": true

}

```
 
### Metadatos del modelo
 
`GET /model/metadata`
 
Devuelve información técnica y descriptiva del modelo entrenado, incluyendo:
 
- nombre y versión del modelo;

- fecha de entrenamiento;

- activos soportados;

- variables utilizadas;

- métricas de evaluación;

- horizonte de predicción;

- advertencia sobre el uso del modelo.
 
### Datos de mercado
 
`GET /market-data/{symbol}`
 
Devuelve la información financiera más reciente disponible para el activo solicitado.
 
Ejemplo de solicitud:
 
```text

GET /market-data/AAPL

```
 
La respuesta incluye:
 
- símbolo del activo;

- fecha del registro;

- precio de cierre;

- retorno diario;

- medias móviles de 5 y 10 días;

- volatilidad de 5 días;

- fuente de los datos.
 
### Generación de predicciones
 
`POST /predict`
 
Genera una predicción sobre la tendencia del precio del activo para el siguiente día.
 
Ejemplo de solicitud:
 
```json

{

  "symbol": "AAPL"

}

```
 
Ejemplo de respuesta:
 
```json

{

  "symbol": "AAPL",

  "prediction": "down",

  "prediction_class": 0,

  "probability_up": 0.4069,

  "model_name": "RandomForestClassifier",

  "model_version": "randomforestclassifier_v1",

  "prediction_horizon": "next_day",

  "data_date": "2025-12-29",

  "data_source": "cached",

  "disclaimer": "Modelo desarrollado con fines educativos. No constituye asesoría financiera."

}

```
 
Los símbolos soportados actualmente son:
 
- `AAPL`

- `MSFT`

- `GOOGL`
 
### Documentación interactiva
 
Con la API en ejecución, FastAPI genera automáticamente documentación interactiva:
 
- Swagger UI: `http://127.0.0.1:8000/docs`

- ReDoc: `http://127.0.0.1:8000/redoc`
 

 ## Pruebas automatizadas
 
El proyecto utiliza `pytest` y `FastAPI TestClient` para validar automáticamente el funcionamiento de los principales endpoints de la API.
 
Para ejecutar las pruebas:
 
```bash

poetry run pytest tests/test_api.py -v

```
 
Actualmente se incluyen pruebas para:
 
- verificar la respuesta HTTP 200 del endpoint `GET /health`;

- validar la estructura de la respuesta del estado del servicio;

- verificar el endpoint `GET /model/metadata`;

- validar la estructura de los metadatos del modelo;

- verificar la consulta de datos mediante `GET /market-data/AAPL`;

- validar la estructura de los datos de mercado;

- verificar la generación de predicciones mediante `POST /predict`;

- validar la estructura de la respuesta de predicción;

- comprobar el rechazo de símbolos no soportados en `POST /predict`;

- comprobar la respuesta HTTP 404 para símbolos no disponibles en los datos de mercado.
 
### Resultado de las pruebas
 
La ejecución actual del conjunto de pruebas obtuvo:
 
```text

10 passed

0 failed

```
 
Las advertencias mostradas durante la ejecución corresponden a dependencias externas y no afectan el resultado de las pruebas ni el funcionamiento actual de la API.
 

 ## Ejecución con Docker
 
El proyecto puede ejecutarse dentro de un contenedor Docker, lo que permite disponer de un entorno aislado y reproducible con las dependencias necesarias para ejecutar la API.
 
### Construir la imagen
 
Desde la raíz del proyecto, ejecutar:
 
```bash

docker build -t financial-api .

```
 
Este comando construye una imagen llamada:
 
```text

financial-api:latest

```
 
La imagen incluye:
 
- Python 3.12;

- dependencias principales del proyecto;

- código fuente de la aplicación;

- modelo entrenado y sus metadatos;

- dataset procesado requerido por la API.
 
### Ejecutar el contenedor
 
Una vez construida la imagen:
 
```bash

docker run --rm -p 8000:8000 financial-api

```
 
La opción `-p 8000:8000` conecta el puerto 8000 del contenedor con el puerto 8000 del equipo local.
 
La API estará disponible en:
 
- API: `http://127.0.0.1:8000`

- Swagger UI: `http://127.0.0.1:8000/docs`

- ReDoc: `http://127.0.0.1:8000/redoc`
 
### Detener el contenedor
 
Si el contenedor se está ejecutando en primer plano, puede detenerse mediante:
 
```text

Ctrl + C

```
 
Durante la validación del proyecto se comprobó correctamente:
 
- la construcción de la imagen `financial-api`;

- el inicio del contenedor;

- la ejecución de FastAPI mediante Uvicorn;

- el acceso a la documentación interactiva;

- la respuesta HTTP 200 del endpoint `GET /health` desde el entorno contenerizado.
 

 ## Resultados del modelo
 
Para respetar el orden temporal de los datos financieros, se realizó una división temporal entre los conjuntos de entrenamiento y prueba.
 
- Fecha de corte: `2024-10-17`

- Registros de entrenamiento: `3.588`

- Registros de prueba: `900`
 
Se entrenaron y compararon dos modelos de clasificación: Random Forest y Logistic Regression.
 
### Random Forest
 
| Métrica | Resultado |

|---|---:|

| Accuracy | 0.5000 |

| Precision | 0.5621 |

| Recall | 0.3862 |

| F1-score | 0.4578 |

| ROC-AUC | 0.5141 |
 
### Logistic Regression
 
| Métrica | Resultado |

|---|---:|

| Accuracy | 0.4911 |

| Precision | 0.5612 |

| Recall | 0.3171 |

| F1-score | 0.4052 |

| ROC-AUC | 0.5036 |
 
El modelo seleccionado fue `RandomForestClassifier`, al obtener el mayor valor de ROC-AUC entre los modelos evaluados.
 
## Limitaciones del modelo
 
Los resultados obtenidos muestran una capacidad predictiva limitada y cercana a una clasificación aleatoria. Esto refleja la dificultad de predecir movimientos diarios del mercado financiero utilizando únicamente información histórica y variables técnicas simples.
 
El objetivo principal del proyecto no es obtener un modelo con alto rendimiento financiero, sino demostrar la implementación de un flujo reproducible de Machine Learning y MLOps que incluye:
 
- ingesta y almacenamiento local de datos;

- ingeniería de variables;

- entrenamiento y evaluación de modelos;

- serialización del modelo;

- almacenamiento de métricas y metadatos;

- generación de predicciones;

- exposición del modelo mediante una API REST;

- validación mediante pruebas automatizadas;

- ejecución reproducible mediante Docker.
 
El desempeño del modelo podría mejorarse en trabajos futuros mediante la incorporación de nuevas fuentes de información, variables adicionales, técnicas de selección de características, ajuste de hiperparámetros y estrategias de validación temporal más avanzadas.
 
## Advertencia de uso
 
Este proyecto fue desarrollado exclusivamente con fines académicos y educativos.
 
Las predicciones generadas por el modelo no constituyen asesoría financiera ni representan una recomendación de compra, venta o conservación de activos. Los resultados no deben utilizarse como único criterio para tomar decisiones de inversión.
 