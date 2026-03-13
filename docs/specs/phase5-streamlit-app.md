# Phase 5 — Streamlit App (MVP Demo)

## Objectif
Application Streamlit qui interagit **exclusivement** avec l'API FastAPI déployée.
3 onglets comme demandé par le prof : contexte, exploration des données, résultats du modèle.

## Exigences du prof
- Interface orientée utilisateur pour explorer les données et les sorties du modèle
- Interagir exclusivement avec l'API déployée
- Proposer plusieurs vues (onglets) : contexte, exploration des données, résultats du modèle

## Architecture

```
Streamlit (:8501) --HTTP--> API FastAPI (:8000) --load--> models/*.pkl
```

Streamlit ne charge JAMAIS de modèle directement. Tout passe par l'API.

## Fichiers à créer

```
src/ui/
  app.py              # Point d'entrée Streamlit
  pages/
    1_Contexte.py      # Onglet 1 : contexte business
    2_Data_Explorer.py # Onglet 2 : exploration des données
    3_Predictions.py   # Onglet 3 : prédiction live via API
  utils.py            # Helper pour appels API
requirements-ui.txt   # Dépendances Streamlit
docker/streamlit/Dockerfile
```

## Onglet 1 : Contexte Business
- Titre du projet, description du problème Rakuten
- Schéma de l'architecture MLOps (image ou mermaid)
- Métriques clés du modèle (accuracy, F1) récupérées depuis `/health` ou affichées en statique
- Liens vers les services (MLflow :5000, Airflow :8080, Grafana :3000)

## Onglet 2 : Data Explorer
- Charger un sample CSV local (data/preprocessed/X_train_clean.csv ou embedded)
- Distribution des catégories (bar chart)
- Exemples de produits par catégorie (tableau filtrable)
- Stats descriptives (nb produits, longueur moyenne texte, etc.)
- Note : les données peuvent être lues localement (montées en volume), pas besoin de passer par l'API pour l'exploration

## Onglet 3 : Prédictions Live
- Formulaire : champ "designation" (obligatoire) + "description" (optionnel)
- Bouton "Prédire" → appel POST /predict via l'API
- Affichage : catégorie prédite, code, confiance (gauge chart ou progress bar)
- Exemples pré-remplis (boutons quick-fill avec des vrais produits Rakuten)
- Historique des prédictions de la session (tableau en bas)

## Configuration API
- URL API : variable d'environnement `API_URL` (default: `http://api:8000` en Docker, `http://localhost:8000` en local)
- Token : variable d'environnement `API_AUTH_TOKEN`

## Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
COPY src/ui/ /app/src/ui/
COPY data/preprocessed/ /app/data/preprocessed/
EXPOSE 8501
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Docker Compose (à ajouter)
```yaml
  streamlit:
    build:
      context: .
      dockerfile: docker/streamlit/Dockerfile
    environment:
      API_URL: http://api:8000
      API_AUTH_TOKEN: ${API_AUTH_TOKEN:-change-me}
    ports:
      - "8501:8501"
    volumes:
      - ./data/preprocessed:/app/data/preprocessed
    depends_on:
      api:
        condition: service_healthy
```

## Dépendances (requirements-ui.txt)
```
streamlit>=1.30.0
requests
pandas
plotly
```

## Style / Wow Effect
- Utiliser st.set_page_config(page_title="Rakuten MLOps", page_icon="🛒", layout="wide")
- Couleurs cohérentes (palette Rakuten : rouge #BF0000)
- Sidebar avec logo/titre du projet
- Animations sur les prédictions (st.balloons() au premier predict, spinners pendant le chargement)
- Metric cards pour les KPIs

## Critères de succès
- [ ] `streamlit run src/ui/app.py` fonctionne en local
- [ ] Les 3 onglets sont navigables
- [ ] Une prédiction via l'API fonctionne dans l'onglet 3
- [ ] Le service tourne dans Docker Compose
- [ ] Visuellement impressionnant pour une soutenance
