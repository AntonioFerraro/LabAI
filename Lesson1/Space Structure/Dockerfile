FROM python:3.10-slim

WORKDIR /app

ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}