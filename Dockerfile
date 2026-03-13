# img python legere
FROM python:3.10-slim

# rep de travail
WORKDIR /app

# maj pip
RUN pip install --no-cache-dir --upgrade pip

# copie reqs
COPY requirements.txt .

# inst deps
RUN pip install --no-cache-dir -r requirements.txt

# cree le dossier models (les artefacts sont montes en volume via
# docker-compose ou generes par le service bootstrap)
RUN mkdir -p /app/models

# copie code source
COPY src/ /app/src/

# port api
EXPOSE 8000

# start api
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]