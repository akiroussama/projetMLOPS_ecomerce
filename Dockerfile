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

# copie modeles (on garde la structure dvc)
COPY models/ /app/models/

# si ton code cherche le pkl a la racine de models on fait un lien
# sinon on laisse la structure dvc standard
COPY models/artifacts/model.pkl /app/models/model.pkl

# copie code source (une seule fois suffit)
COPY src/ /app/src/

# port api
EXPOSE 8000

# start api (on pointe sur main.py de mika)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]