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
    <style>
    .mlops-grid {
        display: grid;
        grid-template-columns: 1fr 40px 1fr 40px 1fr;
        gap: 10px;
        align-items: center;
        justify-items: center;
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    .mlops-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        width: 100%;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }

    .mlops-card:hover {
        border-color: #BF0000;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(191,0,0,0.08);
    }

    .mlops-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .mlops-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.6rem;
    }

    .mlops-port {
        font-family: 'Fira Code', monospace;
        font-size: 0.75rem;
        background: #fee2e2;
        color: #BF0000;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-weight: 600;
    }

    .mlops-arrow {
        font-size: 1.5rem;
        color: #cbd5e1;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .mlops-arrow-text {
        font-size: 0.6rem;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: -4px;
    }

    .api-card {
        border-color: #22c55e;
        border-width: 2px;
        background: #f0fdf4;
    }
    </style>

    <div class="mlops-grid">
        <!-- Row 1 -->
        <div class="mlops-card">
            <div class="mlops-title">🔄 Airflow</div>
            <div class="mlops-desc">Orchestration</div>
            <span class="mlops-port">:8080</span>
        </div>
        <div class="mlops-arrow">➔</div>
        <div class="mlops-card">
            <div class="mlops-title">🧹 Preprocessing</div>
            <div class="mlops-desc">Nettoyage texte</div>
        </div>
        <div class="mlops-arrow">➔</div>
        <div class="mlops-card">
            <div class="mlops-title">⚙️ Training</div>
            <div class="mlops-desc">TF-IDF + SGD</div>
        </div>

        <!-- Row 2 -->
        <div></div>
        <div></div>
        <div></div>
        <div></div>
        <div class="mlops-arrow" style="margin: 5px 0;">⬇</div>

        <!-- Row 3 -->
        <div class="mlops-card">
            <div class="mlops-title">🖥️ Streamlit</div>
            <div class="mlops-desc">Interface Utilisateur</div>
            <span class="mlops-port">:8501</span>
        </div>
        <div class="mlops-arrow">
            <span class="mlops-arrow-text">HTTP</span>
            ➔
        </div>
        <div class="mlops-card api-card">
            <div class="mlops-title">🚀 FastAPI</div>
            <div class="mlops-desc">Serving API REST</div>
            <span class="mlops-port">:8000</span>
        </div>
        <div class="mlops-arrow">⬅</div>
        <div class="mlops-card">
            <div class="mlops-title">📈 MLflow</div>
            <div class="mlops-desc">Experiment Tracking</div>
            <span class="mlops-port">:5000</span>
        </div>

        <!-- Row 4 -->
        <div></div>
        <div></div>
        <div class="mlops-arrow" style="margin: 5px 0;">⬇</div>
        <div></div>
        <div></div>

        <!-- Row 5 -->
        <div></div>
        <div></div>
        <div class="mlops-card">
            <div class="mlops-title">📦 Artifacts</div>
            <div class="mlops-desc">Volume /Models</div>
        </div>
        <div></div>
        <div class="mlops-card">
            <div class="mlops-title">📊 Grafana</div>
            <div class="mlops-desc">Monitoring Métriques</div>
            <span class="mlops-port">:3000</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.5rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:2rem; margin-bottom:0.3rem;">🚀</div>
            <h4 style="margin:0 0 0.3rem 0;">FastAPI</h4>
            <p style="font-size:0.85rem; color:#555; margin:0 0 0.5rem 0;">API de prediction</p>
            <a href="http://localhost:8000/docs" target="_blank"
               style="color:#BF0000; font-weight:600; font-size:0.9rem;">
                localhost:8000/docs ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.5rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:2rem; margin-bottom:0.3rem;">📈</div>
            <h4 style="margin:0 0 0.3rem 0;">MLflow</h4>
            <p style="font-size:0.85rem; color:#555; margin:0 0 0.5rem 0;">Experiment tracking</p>
            <a href="http://localhost:5000" target="_blank"
               style="color:#BF0000; font-weight:600; font-size:0.9rem;">
                localhost:5000 ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.5rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:2rem; margin-bottom:0.3rem;">🔄</div>
            <h4 style="margin:0 0 0.3rem 0;">Airflow</h4>
            <p style="font-size:0.85rem; color:#555; margin:0 0 0.5rem 0;">Orchestration DAGs</p>
            <a href="http://localhost:8080" target="_blank"
               style="color:#BF0000; font-weight:600; font-size:0.9rem;">
                localhost:8080 ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s4:
    st.markdown(
        """
        <div style="background:#fff; border-radius:10px; padding:1.5rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:2rem; margin-bottom:0.3rem;">📊</div>
            <h4 style="margin:0 0 0.3rem 0;">Grafana</h4>
            <p style="font-size:0.85rem; color:#555; margin:0 0 0.5rem 0;">Monitoring</p>
            <a href="http://localhost:3000" target="_blank"
               style="color:#BF0000; font-weight:600; font-size:0.9rem;">
                localhost:3000 ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")
st.caption("Rakuten MLOps Project - Tous les services tournent dans Docker Compose.")
