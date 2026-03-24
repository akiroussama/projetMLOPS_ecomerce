"""
Page 1 - Contexte Business
===========================
Project overview, MLOps architecture diagram, key metrics, service links.
"""

import streamlit as st

st.set_page_config(
    page_title="Contexte | Rakuten MLOps",
    page_icon="R",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">
            📋 Contexte du Projet
        </h1>
        <p style="font-size: 1.05rem; color: #555;">
            Classification automatique de produits e-commerce Rakuten
        </p>
    </div>
    <hr style="border:none; height:2px; background:linear-gradient(90deg,#BF0000,transparent); margin-bottom:1.5rem;">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("## Le Challenge Rakuten")
    st.markdown(
        """
        **Rakuten France** gere des millions de produits sur sa marketplace.
        Chaque produit doit etre associe a la bonne categorie (`prdtypecode`)
        pour garantir une experience de recherche et de navigation optimale.

        **Probleme :** La categorisation manuelle est lente, couteuse et sujette
        aux erreurs humaines. Il faut une solution automatisee.

        **Notre solution :** Un pipeline MLOps complet qui :
        - Pretraite les textes produits (designation + description)
        - Entraine un classifieur supervise (TF-IDF + modele ML)
        - Deploie le modele derriere une API REST securisee
        - Monitore les performances en continu
        """
    )

with col_right:
    st.markdown("## Donnees d'entree")
    st.markdown(
        """
        | Champ | Description |
        |-------|-------------|
        | `designation` | Titre court du produit |
        | `description` | Description longue (HTML nettoyee) |
        | `productid` | Identifiant unique du produit |
        | `imageid` | Identifiant de l'image associee |
        | `prdtypecode` | **Categorie cible** (27 classes) |
        """
    )
    st.markdown("")
    st.info(
        "Le modele utilise uniquement les champs textuels "
        "(**designation** + **description**) pour predire le `prdtypecode`."
    )

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:2rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Architecture diagram (text-based)
# ---------------------------------------------------------------------------
st.markdown("## Architecture MLOps")

st.markdown(
    """
    <div style="text-align: center; margin: 0.5rem 0;">
        <svg viewBox="0 0 900 300" style="max-width: 95%; height: auto;">
            <defs>
                <marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L7,3 z" fill="#64748b" />
                </marker>
                <marker id="arrRed" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L7,3 z" fill="#bf0000" />
                </marker>
            </defs>

            <!-- User -->
            <rect x="10" y="110" width="90" height="60" rx="10" fill="#f3f4f6" stroke="#94a3b8" stroke-width="2" />
            <text x="55" y="135" text-anchor="middle" font-size="22">&#128100;</text>
            <text x="55" y="158" text-anchor="middle" font-size="11" font-weight="600" fill="#64748b">Utilisateur</text>

            <!-- Arrow to Streamlit -->
            <line x1="102" y1="140" x2="140" y2="140" stroke="#64748b" stroke-width="2" marker-end="url(#arr)" />

            <!-- Streamlit -->
            <rect x="145" y="100" width="110" height="80" rx="10" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" />
            <text x="200" y="130" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">Streamlit</text>
            <text x="200" y="148" text-anchor="middle" font-size="10" fill="#92400e">:8501</text>
            <text x="200" y="165" text-anchor="middle" font-size="9" fill="#64748b">UI + Predictions</text>

            <!-- Arrow Streamlit to API -->
            <line x1="257" y1="140" x2="295" y2="140" stroke="#bf0000" stroke-width="2.5" marker-end="url(#arrRed)" />

            <!-- API -->
            <rect x="300" y="90" width="120" height="100" rx="10" fill="#fef2f2" stroke="#bf0000" stroke-width="2.5" />
            <text x="360" y="118" text-anchor="middle" font-size="14" font-weight="800" fill="#bf0000">FastAPI</text>
            <text x="360" y="138" text-anchor="middle" font-size="10" fill="#991b1b">:8000 (ext 8200)</text>
            <text x="360" y="152" text-anchor="middle" font-size="9" fill="#64748b">/predict /health</text>
            <text x="360" y="165" text-anchor="middle" font-size="9" fill="#64748b">/metrics /stats</text>

            <!-- Arrow API to Models volume -->
            <line x1="360" y1="192" x2="360" y2="235" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

            <!-- Models Volume -->
            <rect x="315" y="240" width="90" height="45" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="5,3" />
            <text x="360" y="260" text-anchor="middle" font-size="10" font-weight="600" fill="#4338ca">models/</text>
            <text x="360" y="275" text-anchor="middle" font-size="8" fill="#64748b">Volume partage</text>

            <!-- Arrow API to Prometheus -->
            <line x1="422" y1="115" x2="470" y2="65" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

            <!-- Prometheus -->
            <rect x="475" y="30" width="110" height="65" rx="10" fill="#fff7ed" stroke="#f97316" stroke-width="2" />
            <text x="530" y="55" text-anchor="middle" font-size="13" font-weight="700" fill="#ea580c">Prometheus</text>
            <text x="530" y="72" text-anchor="middle" font-size="10" fill="#9a3412">:9090</text>
            <text x="530" y="86" text-anchor="middle" font-size="9" fill="#64748b">Scrape /metrics</text>

            <!-- Arrow Prometheus to Grafana -->
            <line x1="587" y1="62" x2="630" y2="62" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

            <!-- Grafana -->
            <rect x="635" y="30" width="100" height="65" rx="10" fill="#ecfdf5" stroke="#10b981" stroke-width="2" />
            <text x="685" y="55" text-anchor="middle" font-size="13" font-weight="700" fill="#059669">Grafana</text>
            <text x="685" y="72" text-anchor="middle" font-size="10" fill="#047857">:3000</text>
            <text x="685" y="86" text-anchor="middle" font-size="9" fill="#64748b">Dashboards</text>

            <!-- Arrow to MLflow -->
            <line x1="422" y1="160" x2="470" y2="180" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

            <!-- MLflow -->
            <rect x="475" y="155" width="110" height="65" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2" />
            <text x="530" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">MLflow</text>
            <text x="530" y="197" text-anchor="middle" font-size="10" fill="#1e40af">:5000</text>
            <text x="530" y="211" text-anchor="middle" font-size="9" fill="#64748b">Tracking Server</text>

            <!-- Arrow Airflow to MLflow -->
            <line x1="710" y1="190" x2="587" y2="190" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />

            <!-- Airflow -->
            <rect x="715" y="140" width="120" height="100" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="2" />
            <text x="775" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#6d28d9">Airflow</text>
            <text x="775" y="188" text-anchor="middle" font-size="10" fill="#5b21b6">:8080 (ext 8280)</text>
            <text x="775" y="205" text-anchor="middle" font-size="9" fill="#64748b">Scheduler</text>
            <text x="775" y="220" text-anchor="middle" font-size="9" fill="#64748b">Webserver</text>

            <!-- Postgres -->
            <rect x="740" y="255" width="80" height="40" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5" />
            <text x="780" y="275" text-anchor="middle" font-size="10" font-weight="600" fill="#4338ca">Postgres</text>
            <text x="780" y="288" text-anchor="middle" font-size="8" fill="#64748b">Airflow DB</text>

            <!-- Arrow Airflow to Postgres -->
            <line x1="775" y1="242" x2="780" y2="253" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)" />
        </svg>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin-top:0.5rem;">
        <div style="background:#eff6ff; border-left:4px solid #2563eb; border-radius:8px; padding:1rem;">
            <div style="font-weight:700; color:#1e40af; margin-bottom:0.3rem;">🐳 Docker Compose</div>
            <p style="font-size:0.88rem; color:#555; margin:0;">8+ services orchestres. Un seul <code>docker compose up</code> pour tout demarrer.</p>
        </div>
        <div style="background:#f0fdf4; border-left:4px solid #22c55e; border-radius:8px; padding:1rem;">
            <div style="font-weight:700; color:#15803d; margin-bottom:0.3rem;">📦 Volumes partages</div>
            <p style="font-size:0.88rem; color:#555; margin:0;">Artefacts modele accessibles par API, Trainer, et Airflow via le volume <code>./models</code>.</p>
        </div>
        <div style="background:#fffbeb; border-left:4px solid #f59e0b; border-radius:8px; padding:1rem;">
            <div style="font-weight:700; color:#b45309; margin-bottom:0.3rem;">🔗 Integration fluide</div>
            <p style="font-size:0.88rem; color:#555; margin:0;">Payload API hybride pour concatener a la volee <code>designation</code> et <code>description</code> de Streamlit.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:2rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.markdown("## Metriques Cles du Modele")

import sys  # noqa: E402
import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils import check_api_health

    with st.spinner("Interrogation de l'API..."):
        health = check_api_health()
    api_status = health.get("status", "unknown")
    model_name = health.get("model_name", "N/A")
    model_loaded = health.get("model_loaded", False)
    detail = health.get("detail", "")
except Exception:
    api_status = "offline"
    model_name = "N/A"
    model_loaded = False
    detail = "API non joignable"

m1, m2, m3, m4 = st.columns(4)

with m1:
    status_emoji = "✅" if api_status == "ok" else ("⚠️" if api_status == "degraded" else "❌")
    st.metric("Statut API", f"{status_emoji} {api_status.upper()}")

with m2:
    st.metric("Modele", model_name)

with m3:
    st.metric("Modele charge", "Oui" if model_loaded else "Non")

with m4:
    st.metric("Categories", "27 classes")

st.markdown("")

if api_status == "ok":
    st.success(f"Le modele est operationnel. {detail}")
elif api_status == "degraded":
    st.warning(f"API en mode degrade : {detail}")
else:
    st.info(
        "L'API n'est pas joignable pour le moment. Les metriques ci-dessus "
        "affichent des valeurs par defaut. Demarrez le service API pour voir "
        "les metriques en direct."
    )

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:2rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Service links
# ---------------------------------------------------------------------------
st.markdown("## Acces aux Services")

_HOST = os.getenv("SERVICES_BASE_URL", "http://rakuten-mlops.duckdns.org")

_SERVICES = [
    {"icon": "🚀", "name": "FastAPI",    "desc": "API de prediction",    "url": f"{_HOST}:8200/docs", "label": "Swagger UI"},
    {"icon": "📈", "name": "MLflow",     "desc": "Experiment tracking",  "url": f"{_HOST}:5000",      "label": "Tracking UI"},
    {"icon": "🔄", "name": "Airflow",    "desc": "Orchestration DAGs",   "url": f"{_HOST}:8280",      "label": "Webserver"},
    {"icon": "📊", "name": "Grafana",    "desc": "Dashboards monitoring","url": f"{_HOST}:3000",      "label": "Dashboard"},
    {"icon": "🔥", "name": "Prometheus", "desc": "Collecte de metriques","url": f"{_HOST}:9090",      "label": "Targets"},
]

_cols = st.columns(len(_SERVICES))
for col, svc in zip(_cols, _SERVICES):
    with col:
        st.markdown(
            f"""
            <div style="background:#fff; border-radius:10px; padding:1.2rem;
                        border:1px solid #e0e0e0; text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:2rem; margin-bottom:0.3rem;">{svc["icon"]}</div>
                <h4 style="margin:0 0 0.3rem 0;">{svc["name"]}</h4>
                <p style="font-size:0.82rem; color:#555; margin:0 0 0.5rem 0;">{svc["desc"]}</p>
                <a href="{svc["url"]}" target="_blank"
                   style="color:#BF0000; font-weight:600; font-size:0.85rem;">
                    {svc["label"]} ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.caption("Rakuten MLOps Project - Tous les services tournent dans Docker Compose.")
