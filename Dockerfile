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

# copie tout le dossier models (inclut artifacts et label_mapping.json)
# plus besoin de copies individuelles, le dossier src s'occupe du reste
COPY models/ /app/models/

# copie code source
COPY src/ /app/src/

# port api
EXPOSE 8000

# start api
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]