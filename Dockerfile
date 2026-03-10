
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

# copie code et modeles
COPY src/ /app/src/
COPY models/artifacts/model.pkl /app/models/model.pkl
COPY models/label_mapping.json /app/models/label_mapping.json

# port api
EXPOSE 8000

# variable env pr token


# start api
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]