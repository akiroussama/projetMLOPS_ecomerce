#!/usr/bin/env bash
# =============================================================================
#  server_deploy.sh — Déploiement complet MLOps Rakuten sur VPS Ubuntu
#
#  Usage : ssh root@<IP> 'bash -s' < scripts/server_deploy.sh
#  Ou    : copier ce fichier sur le VPS, puis : bash server_deploy.sh
#
#  Pré-requis : Ubuntu 22.04 ou 24.04, accès root, 8 GB RAM minimum
# =============================================================================
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/akiroussama/projetMLOPS_ecomerce.git"
APP_DIR="/opt/rakuten-mlops"
API_AUTH_TOKEN="rakuten-soutenance-2024"
SWAP_SIZE_GB=4
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${BLUE}[→]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MLOps Rakuten — Déploiement VPS automatique        ${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo ""

# ─── 1. Swap (évite les OOM pendant le build Docker) ─────────────────────────
info "Étape 1/7 — Ajout de ${SWAP_SIZE_GB}GB de swap..."
if [ ! -f /swapfile ]; then
    fallocate -l "${SWAP_SIZE_GB}G" /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    log "Swap ${SWAP_SIZE_GB}GB ajouté"
else
    warn "Swap déjà configuré, on passe"
fi

# ─── 2. Dépendances système ───────────────────────────────────────────────────
info "Étape 2/7 — Installation des dépendances système..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    ca-certificates curl git gnupg ufw lsb-release

# ─── 3. Docker ───────────────────────────────────────────────────────────────
info "Étape 3/7 — Installation de Docker..."
if ! command -v docker &> /dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    log "Docker installé"
else
    warn "Docker déjà installé, on passe"
fi

# ─── 4. Firewall — ouvrir les ports nécessaires ──────────────────────────────
info "Étape 4/7 — Configuration du firewall..."
ufw allow OpenSSH > /dev/null
ufw allow 8501/tcp comment "Streamlit" > /dev/null
ufw allow 5000/tcp comment "MLflow" > /dev/null
ufw allow 8280/tcp comment "Airflow" > /dev/null
ufw allow 3000/tcp comment "Grafana" > /dev/null
ufw allow 8200/tcp comment "API Swagger" > /dev/null
ufw allow 9090/tcp comment "Prometheus" > /dev/null
ufw --force enable > /dev/null
log "Firewall configuré (SSH + 6 ports applicatifs)"

# ─── 5. Clone du dépôt ───────────────────────────────────────────────────────
info "Étape 5/7 — Clonage du dépôt..."
if [ -d "$APP_DIR/.git" ]; then
    warn "Dépôt déjà présent — mise à jour (git pull)..."
    cd "$APP_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi
log "Dépôt prêt dans $APP_DIR"

# ─── 6. Configuration ────────────────────────────────────────────────────────
info "Configuration de l'environnement..."
cat > "$APP_DIR/.env" << EOF
API_AUTH_TOKEN=${API_AUTH_TOKEN}
AIRFLOW_UID=50000
EOF

# Répertoires dont Airflow a besoin en écriture
mkdir -p "$APP_DIR/orchestration/logs"
mkdir -p "$APP_DIR/orchestration/dags"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/data/preprocessed"
chmod -R 777 "$APP_DIR/orchestration/logs"
chmod -R 777 "$APP_DIR/data"
chmod -R 777 "$APP_DIR/models"
log "Environnement configuré"

# ─── 7. Bootstrap (téléchargement données + entraînement modèle) ─────────────
info "Étape 6/7 — Bootstrap : téléchargement des données + entraînement..."
warn "Cette étape peut prendre 15-25 minutes (build Docker + train)..."
cd "$APP_DIR"

# Démarrer MLflow d'abord (le bootstrap en a besoin)
docker compose up -d mlflow
sleep 10

# Lancer le bootstrap
docker compose --profile bootstrap up bootstrap --build --exit-code-from bootstrap
log "Bootstrap terminé — modèles entraînés et dans ./models/"

# ─── 8. Lancement de tous les services ───────────────────────────────────────
info "Étape 7/7 — Démarrage de tous les services..."
docker compose up -d --build

# Attendre que les services soient prêts
info "Attente du démarrage des services (45 secondes)..."
sleep 45

# ─── Résumé final ────────────────────────────────────────────────────────────
SERVER_IP=$(curl -4 -s ifconfig.me 2>/dev/null || curl -4 -s icanhazip.com 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   DÉPLOIEMENT TERMINÉ — Services disponibles         ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Streamlit${NC}   →  http://${SERVER_IP}:8501"
echo -e "  ${BLUE}MLflow${NC}      →  http://${SERVER_IP}:5000"
echo -e "  ${BLUE}Airflow${NC}     →  http://${SERVER_IP}:8280   (airflow / airflow)"
echo -e "  ${BLUE}Grafana${NC}     →  http://${SERVER_IP}:3000   (admin / admin)"
echo -e "  ${BLUE}Swagger${NC}     →  http://${SERVER_IP}:8200/docs"
echo -e "  ${BLUE}Prometheus${NC}  →  http://${SERVER_IP}:9090"
echo ""
echo -e "  ${YELLOW}API Token${NC}   :  ${API_AUTH_TOKEN}"
echo ""
echo -e "  Pour voir les logs : cd ${APP_DIR} && docker compose logs -f"
echo -e "  Pour arrêter       : cd ${APP_DIR} && docker compose down"
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"

# Vérification rapide de l'API
sleep 5
if curl -sf "http://localhost:8200/health" > /dev/null 2>&1; then
    log "API health check : OK"
else
    warn "API pas encore prête — attendre encore 1-2 minutes puis rafraîchir"
fi
