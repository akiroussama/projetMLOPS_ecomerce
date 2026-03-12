
FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-api.txt

COPY src/ /app/src/
COPY setup.py README.md /app/

RUN mkdir -p /app/models

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
