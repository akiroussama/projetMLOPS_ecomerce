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

# ON RECOPIE LES MODELES (indispensable pour que le conteneur soit intelligent)
COPY models/ /app/models/

# copie code source
COPY src/ /app/src/

# port api
EXPOSE 8000

# ON POINTE SUR main.py (celui qui a nos fix de compatibilité)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]