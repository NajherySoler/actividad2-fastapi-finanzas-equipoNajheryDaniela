FROM python:3.12-slim
 
WORKDIR /app

ENV PYTHONPATH=/app/src
 
RUN pip install --no-cache-dir poetry
 
COPY pyproject.toml poetry.lock ./
 
RUN poetry config virtualenvs.create false \
&& poetry install --only main --no-root --no-interaction --no-ansi
 
COPY src ./src

COPY artifacts ./artifacts

COPY data/processed ./data/processed
 
EXPOSE 8000
 
CMD ["uvicorn", "financial_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
 