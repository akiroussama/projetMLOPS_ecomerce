# Checklist Demo Oussama — Soutenance MLOps 24/03/2026

## T-15 min avant de passer : lancer le script

```bash
cd D:\workspace\projetMLOPS_ecomerce
python scripts/generate_grafana_traffic.py
```

Durée ~2 min. Quand tu vois "GRAFANA EST PRÊT ✓", passe à l'étape suivante.

---

## T-10 min : ouvrir les 5 onglets dans cet ordre précis

| # | Service | URL | État attendu |
|---|---------|-----|--------------|
| 1 | Streamlit | http://rakuten-mlops.duckdns.org:8501 | Page Accueil, bandeau vert visible |
| 2 | Grafana | http://rakuten-mlops.duckdns.org:3000/d/rakuten-api-monitoring/rakuten-api-monitoring | Dashboard chargé, 4 panneaux avec courbes |
| 3 | MLflow | http://rakuten-mlops.duckdns.org:5000/#/experiments/1 | Liste des 26 runs visible |
| 4 | Airflow | http://rakuten-mlops.duckdns.org:8280/dags/rakuten_weekly_retraining/grid | DAG grid view, cases vertes |
| 5 | Swagger | http://rakuten-mlops.duckdns.org:8200/docs | Cadenas visible sur /predict |

> **Ordre important** : Grafana en 2e (après le script). MLflow direct sur l'expérience (pas la home). Airflow direct sur le grid (pas la liste des DAGs).

---

## Pendant la démo : ordre des onglets

1. **Onglet 1 (Streamlit)** → Accueil + bandeau vert → clic Predictions
2. **Onglet 1 (Streamlit)** → Prediction FIFA 24 → Prediction Harry Potter
3. **Onglet 5 (Swagger)** → 5 secondes, pointer le cadenas
4. **Onglet 2 (Grafana)** → 4 panneaux, pointer les courbes (rafraîchit auto)
5. **Onglet 3 (MLflow)** → 26 runs "airflow-retraining"
6. **Onglet 4 (Airflow)** → 6 tâches vertes

---

## Phrase clé à retenir si Grafana plante

> "Prometheus scrape notre endpoint `/metrics` toutes les 5 secondes. On voit ici les X000 requêtes 2xx traitées depuis le déploiement — 100% de succès."

## Si MLflow est lent

> Ne pas attendre. Dire : "MLflow a enregistré nos 26 runs d'entraînement automatiques — on les retrouve ici, tous déclenchés par Airflow."

---

## Timing cible : 5 min

| Séquence | Durée |
|----------|-------|
| Intro + Streamlit Accueil | 40s |
| Prédictions (×2) | 1min30 |
| Swagger (éclair) | 20s |
| Grafana | 1min |
| MLflow | 45s |
| Airflow | 45s |
| **Total** | **~5min** |
