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