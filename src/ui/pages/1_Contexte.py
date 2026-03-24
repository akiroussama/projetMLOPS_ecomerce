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

_card = """
<div style="background:{bg}; border:2px solid {border}; border-radius:10px;
            padding:1rem; text-align:center; height:100%;">
    <div style="font-size:1.05rem; font-weight:700; color:#0f172a; margin-bottom:0.2rem;">{title}</div>
    <div style="font-size:0.8rem; color:#64748b; margin-bottom:0.6rem;">{desc}</div>
    {port_html}
</div>
"""

def _port(p):
    return f'<span style="font-size:0.75rem; background:#fee2e2; color:#BF0000; padding:0.2rem 0.5rem; border-radius:12px; font-weight:600;">{p}</span>' if p else ""

def _arrow(label=""):
    lbl = f'<div style="font-size:0.6rem; color:#94a3b8; font-weight:700;">{label}</div>' if label else ""
    return f'<div style="text-align:center; color:#cbd5e1; font-size:1.4rem; line-height:1;">{lbl}➔</div>'

def _card(title, desc, port="", bg="#f8fafc", border="#e2e8f0"):
    p = _port(port)
    return f"""<div style="background:{bg}; border:2px solid {border}; border-radius:10px;
        padding:1rem; text-align:center;">
        <div style="font-size:1.05rem; font-weight:700; color:#0f172a; margin-bottom:0.2rem;">{title}</div>
        <div style="font-size:0.8rem; color:#64748b; margin-bottom:0.6rem;">{desc}</div>
        {p}
    </div>"""

# Row 1 : Airflow → Preprocessing → Training
r1a, r1b, r1c, r1d, r1e = st.columns([4, 1, 4, 1, 4])
with r1a:
    st.markdown(_card("🔄 Airflow", "Orchestration", ":8280"), unsafe_allow_html=True)
with r1b:
    st.markdown(_arrow(), unsafe_allow_html=True)
with r1c:
    st.markdown(_card("🧹 Preprocessing", "Nettoyage texte"), unsafe_allow_html=True)
with r1d:
    st.markdown(_arrow(), unsafe_allow_html=True)
with r1e:
    st.markdown(_card("⚙️ Training", "TF-IDF + SGD"), unsafe_allow_html=True)

# Down arrow below Training
_, _, _, _, col_down1 = st.columns([4, 1, 4, 1, 4])
with col_down1:
    st.markdown('<div style="text-align:center; color:#cbd5e1; font-size:1.4rem;">⬇</div>', unsafe_allow_html=True)

# Row 2 : Streamlit → FastAPI ← MLflow
r2a, r2b, r2c, r2d, r2e = st.columns([4, 1, 4, 1, 4])
with r2a:
    st.markdown(_card("🖥️ Streamlit", "Interface Utilisateur", ":8501"), unsafe_allow_html=True)
with r2b:
    st.markdown(_arrow("HTTP"), unsafe_allow_html=True)
with r2c:
    st.markdown(_card("🚀 FastAPI", "Serving API REST", ":8200", bg="#f0fdf4", border="#22c55e"), unsafe_allow_html=True)
with r2d:
    st.markdown('<div style="text-align:center; color:#cbd5e1; font-size:1.4rem;">⬅</div>', unsafe_allow_html=True)
with r2e:
    st.markdown(_card("📈 MLflow", "Experiment Tracking", ":5000"), unsafe_allow_html=True)

# Down arrow below FastAPI
_, _, col_down2, _, _ = st.columns([4, 1, 4, 1, 4])
with col_down2:
    st.markdown('<div style="text-align:center; color:#cbd5e1; font-size:1.4rem;">⬇</div>', unsafe_allow_html=True)

# Row 3 : Prometheus → Artifacts ← Grafana
r3a, r3b, r3c, r3d, r3e = st.columns([4, 1, 4, 1, 4])
with r3a:
    st.markdown(_card("🔥 Prometheus", "Collecte metriques", ":9090"), unsafe_allow_html=True)
with r3b:
    st.markdown(_arrow(), unsafe_allow_html=True)
with r3c:
    st.markdown(_card("📦 Artifacts", "Volume /Models"), unsafe_allow_html=True)
with r3d:
    st.markdown('<div style="text-align:center; color:#cbd5e1; font-size:1.4rem;">⬅</div>', unsafe_allow_html=True)
with r3e:
    st.markdown(_card("📊 Grafana", "Monitoring Dashboard", ":3000"), unsafe_allow_html=True)

st.markdown("")

# Architecture explanation columns
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.2rem;
                    border-left:4px solid #BF0000; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="margin:0 0 0.5rem 0;">🔄 Orchestration</h4>
            <p style="font-size:0.88rem; color:#555; margin:0;">
                <b>Apache Airflow</b> orchestre le pipeline complet :
                ingestion, preprocessing, training et evaluation.
                Les DAGs s'executent automatiquement.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.2rem;
                    border-left:4px solid #BF0000; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="margin:0 0 0.5rem 0;">🚀 Serving</h4>
            <p style="font-size:0.88rem; color:#555; margin:0;">
                <b>FastAPI</b> sert le modele via une API REST securisee
                (Bearer token). Endpoint POST /predict pour la classification
                en temps reel.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.2rem;
                    border-left:4px solid #BF0000; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="margin:0 0 0.5rem 0;">📈 Tracking & Monitoring</h4>
            <p style="font-size:0.88rem; color:#555; margin:0;">
                <b>MLflow</b> enregistre les experiences, metriques et artefacts.
                <b>Grafana</b> fournit des dashboards de monitoring en temps reel.
            </p>
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
