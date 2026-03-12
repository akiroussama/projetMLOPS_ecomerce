FROM python:3.10-slim

WORKDIR /app

COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-pipeline.txt

COPY src/ /app/src/
COPY setup.py README.md /app/

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main", "--help"]
