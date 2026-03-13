# Phase 4b — CI/CD GitHub Actions (MVP Demo)

## Objectif
Pipeline CI/CD complète : lint → tests → build Docker → push image.
Wow effect : montrer un pipeline vert sur GitHub avec toutes les étapes.

## État actuel
`.github/workflows/python-app.yml` existe déjà avec :
- Trigger sur push main + PR
- Flake8 lint
- pytest

## Ce qui manque (demandé par le prof)
- Builder et publier les images Docker (Docker build & push)

## Fichier à modifier

```
.github/workflows/python-app.yml   # Enrichir le workflow existant
```

## Workflow cible

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install flake8
      - run: flake8 src/ --max-line-length 127

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt pytest pytest-cov httpx
      - run: pytest --cov=src/api --cov-fail-under=80

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        if: github.event_name == 'push'
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name == 'push' }}
          tags: ghcr.io/${{ github.repository }}/api:latest
```

## Points clés
- **ghcr.io** (GitHub Container Registry) : gratuit, pas besoin de Docker Hub
- Le push d'image ne se fait que sur `push` vers main (pas sur les PR)
- `GITHUB_TOKEN` est automatique, pas besoin de configurer de secrets
- Le job `docker-build` dépend de `test` qui dépend de `lint` → pipeline séquentielle visible

## Scénario de démo
1. Montrer l'onglet Actions sur GitHub
2. Montrer un pipeline vert avec les 3 jobs : lint → test → docker-build
3. "Notre CI/CD vérifie la qualité du code, lance les tests, et publie automatiquement l'image Docker"
4. Optionnel : montrer l'image publiée dans ghcr.io

## Critères de succès
- [ ] Le workflow passe au vert sur GitHub
- [ ] Les 3 jobs (lint, test, docker-build) sont visibles séparément
- [ ] L'image Docker est publiée sur ghcr.io (sur push main)
