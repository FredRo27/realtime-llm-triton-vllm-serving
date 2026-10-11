FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server_matrix.py .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "server_matrix:app", "--host", "0.0.0.0", "--port", "8080", "--ws-ping-interval", "20", "--ws-ping-timeout", "20"]
