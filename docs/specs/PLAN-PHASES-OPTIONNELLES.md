# Plan Phases Optionnelles — MVP "Wow Effect" Soutenance

## Contexte
- Phases 1-3 (obligatoires) : DONE
- Soutenance : 24 mars 2026 (20 min présentation + 10 min Q&A)
- Le jury ne voit PAS le code, uniquement la démo
- Tout est développé par Claude Code → parallélisation maximale

## Estimation par workstream

| # | Spec | Impact démo | Complexité | Dépendances | Fichier spec |
|---|------|-------------|-----------|-------------|--------------|
| 1 | **Streamlit App** | ★★★★★ | Moyenne | API doit tourner | `phase5-streamlit-app.md` |
| 2 | **Grafana + Prometheus** | ★★★★☆ | Moyenne | API doit exposer /metrics | `phase4-monitoring-grafana.md` |
| 3 | **CI/CD GitHub Actions** | ★★★☆☆ | Faible | Aucune | `phase4-cicd-github-actions.md` |
| 4 | **Evidently Drift** | ★★★☆☆ | Faible | Données preprocessed | `phase4-evidently-drift.md` |

## Ordre d'exécution recommandé

```
Parallèle 1 (indépendants) :
  ├── [Agent A] Streamlit App (phase5-streamlit-app.md)
  ├── [Agent B] CI/CD GitHub Actions (phase4-cicd-github-actions.md)
  └── [Agent C] Evidently Drift Report (phase4-evidently-drift.md)

Séquentiel après Agent A :
  └── [Agent D] Grafana + Prometheus (phase4-monitoring-grafana.md)
       (car modifie l'API + docker-compose, mieux de ne pas paralléliser
        avec Streamlit qui touche aussi docker-compose)
```

## Pourquoi cet ordre

1. **Streamlit** est le composant le plus impactant pour la démo — il montre le produit final.
2. **CI/CD** et **Evidently** sont indépendants de tout le reste et simples.
3. **Grafana** touche `docker-compose.yml` et `src/api/app.py` — il vaut mieux le faire après Streamlit pour éviter les conflits de merge dans ces fichiers partagés.

## Modifications par fichier (matrice de conflits)

| Fichier | Streamlit | Grafana | CI/CD | Evidently |
|---------|-----------|---------|-------|-----------|
| `docker-compose.yml` | AJOUTE service | AJOUTE 2 services | - | - |
| `src/api/app.py` | - | MODIFIE (metrics) | - | - |
| `requirements.txt` | - | AJOUTE 1 dep | - | - |
| `.github/workflows/python-app.yml` | - | - | REMPLACE | - |
| Nouveaux fichiers | `src/ui/*`, `docker/streamlit/` | `docker/monitoring/*` | - | `src/monitoring/*` |

→ Streamlit et Grafana touchent tous les deux docker-compose.yml → à séquencer.
→ CI/CD et Evidently sont 100% indépendants → parallélisables avec tout.

## Script de démo 20 min (Demo Machine)

### Acte 1 : Le Problème (3 min) — Slides
"Rakuten reçoit des millions de produits. Comment les classifier automatiquement ?"

### Acte 2 : Les Données (2 min) — Streamlit "Data Explorer"
Ouvrir Streamlit → onglet Data Explorer
Montrer la distribution des catégories, exemples de produits

### Acte 3 : Le Modèle (3 min) — MLflow UI
Ouvrir MLflow :5000
Montrer les runs, comparer les métriques (accuracy, F1)
"On log automatiquement chaque entraînement"

### Acte 4 : Prédiction Live (2 min) — Streamlit "Predict"
Onglet Predictions dans Streamlit
Taper "Console Sony PS5 avec manette DualSense" → catégorie prédite
Taper "Livre de cuisine française" → autre catégorie
WOW MOMENT : la prédiction apparaît en temps réel

### Acte 5 : L'Orchestration (3 min) — Airflow UI
Ouvrir Airflow :8080
Montrer le DAG avec les 5 tâches
Trigger manuel → les tâches passent au vert
"Chaque lundi, le modèle se ré-entraîne automatiquement"

### Acte 6 : L'Infrastructure (2 min) — Slide architecture
Schéma : Streamlit → API → Model ← Airflow ← S3
Docker Compose : 8 services qui communiquent
"Un seul `docker compose up` lance toute la stack"

### Acte 7 : Le Monitoring (2 min) — Grafana
Ouvrir Grafana :3000
Montrer les graphes de latence et requêtes
"Les prédictions qu'on vient de faire apparaissent ici en temps réel"

### Acte 8 : CI/CD + Drift (1 min) — GitHub + HTML
Montrer le pipeline vert sur GitHub Actions
Montrer le rapport Evidently de drift

### Acte 9 : Conclusions (2 min) — Slides
Résultats business, limites, next steps
