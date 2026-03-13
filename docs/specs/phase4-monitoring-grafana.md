# Phase 4a — Monitoring Prometheus + Grafana (MVP Demo)

## Objectif
Dashboard Grafana qui montre les métriques de l'API en temps réel.
Wow effect : pendant la démo, on fait des prédictions dans Streamlit et on voit les graphes bouger dans Grafana.

## Architecture

```
API FastAPI (:8000) --/metrics--> Prometheus (:9090) --scrape--> Grafana (:3000)
```

## Fichiers à créer

```
docker/monitoring/
  prometheus.yml          # Config scrape
  grafana/
    provisioning/
      datasources/
        prometheus.yml    # Auto-provision datasource
      dashboards/
        dashboard.yml     # Auto-provision dashboard provider
    dashboards/
      api-monitoring.json # Dashboard pré-configuré
```

## Modifications à faire

### 1. Ajouter prometheus-fastapi-instrumentator à l'API

Dans `src/api/app.py`, ajouter :
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Ceci expose automatiquement `/metrics` avec :
- `http_requests_total` (par endpoint, méthode, status)
- `http_request_duration_seconds` (histogramme de latence)
- `http_requests_in_progress`

### 2. Ajouter les dépendances

Dans `requirements.txt` ajouter :
```
prometheus-fastapi-instrumentator
```

### 3. prometheus.yml
```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']
```

### 4. Dashboard Grafana (api-monitoring.json)
Dashboard pré-provisionné avec 4 panels :

| Panel | Type | Métrique | Wow effect |
|-------|------|----------|-----------|
| Requêtes/sec | Time series | rate(http_requests_total[1m]) | Graphe qui monte pendant la démo |
| Latence P50/P95 | Time series | histogram_quantile(0.95, http_request_duration_seconds) | Montre la performance |
| Status codes | Pie chart | http_requests_total par status | Vert = 200, rouge = erreurs |
| Prédictions en cours | Stat | http_requests_in_progress | Compteur live |

### 5. Docker Compose (à ajouter)
```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      api:
        condition: service_healthy

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - ./docker/monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./docker/monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
```

Note : `GF_AUTH_ANONYMOUS_ENABLED=true` permet d'accéder au dashboard sans login pendant la démo.

## Scénario de démo
1. Ouvrir Grafana :3000 → dashboard "API Monitoring"
2. Les graphes sont à plat (pas de trafic)
3. Aller sur Streamlit et faire 5-6 prédictions
4. Revenir sur Grafana → les graphes montent en temps réel
5. "Comme vous pouvez le voir, notre monitoring capture chaque requête en temps réel"

## Critères de succès
- [ ] `/metrics` retourne des métriques Prometheus sur l'API
- [ ] Prometheus scrape l'API toutes les 5s
- [ ] Grafana affiche le dashboard auto-provisionné
- [ ] Le dashboard montre du mouvement quand on fait des prédictions
- [ ] Pas besoin de login Grafana pour la démo
