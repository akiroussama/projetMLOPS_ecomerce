"""
Rakuten MLOps - Streamlit Dashboard
====================================
Main entry point for the Streamlit application.
Provides navigation across three tabs: Contexte, Data Explorer, Predictions.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rakuten MLOps",
    page_icon="\ud83d\uded2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global custom CSS  - Rakuten red theme (#BF0000)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Root colour variables ---------- */
    :root {
        --rakuten-red: #BF0000;
        --rakuten-dark: #8B0000;
        --rakuten-light: #FFE5E5;
        --card-bg: #FFFFFF;
        --card-border: #E0E0E0;
        --text-primary: #1A1A2E;
        --text-secondary: #555555;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #BF0000 0%, #8B0000 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown a {
        color: #FFD700 !important;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--rakuten-red);
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--rakuten-red) !important;
        font-weight: 700 !important;
    }

    /* ---------- Headers ---------- */
    h1, h2 {
        color: var(--rakuten-red) !important;
    }
    h3 {
        color: var(--text-primary) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(191,0,0,0.25);
    }

    /* ---------- Utility classes ---------- */
    .rakuten-badge {
        display: inline-block;
        background: var(--rakuten-red);
        color: white !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, var(--rakuten-red), transparent);
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
            <div style="font-size: 3rem;">🛒</div>
            <h1 style="margin:0; font-size:1.6rem; color:white !important;">
                Rakuten MLOps
            </h1>
            <p style="margin:4px 0 0 0; font-size:0.85rem; opacity:0.85;">
                Classification de produits e-commerce
            </p>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2);">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    st.markdown(
        """
        Utilisez le menu ci-dessus pour naviguer entre les pages :

        - **Contexte** - Presentation du projet
        - **Data Explorer** - Exploration des donnees
        - **Predictions** - Predictions en temps reel
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; opacity: 0.7; text-align: center;">
            MLOps Project &bull; 2025-2026<br>
            Streamlit + FastAPI + MLflow
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.3rem;">
            🛒 Rakuten MLOps Dashboard
        </h1>
        <p style="font-size: 1.15rem; color: #555; max-width: 700px; margin: 0 auto;">
            Classification automatique de produits e-commerce
            avec pipeline MLOps complet
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Quick-access cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="background:white; border-radius:12px; padding:2rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📋</div>
            <h3 style="margin:0 0 0.5rem 0;">Contexte</h3>
            <p style="color:#555; font-size:0.9rem;">
                Decouvrez le projet, l'architecture MLOps et les metriques cles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="background:white; border-radius:12px; padding:2rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📊</div>
            <h3 style="margin:0 0 0.5rem 0;">Data Explorer</h3>
            <p style="color:#555; font-size:0.9rem;">
                Explorez les donnees d'entrainement, distributions et statistiques.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="background:white; border-radius:12px; padding:2rem;
                    border:1px solid #e0e0e0; text-align:center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔮</div>
            <h3 style="margin:0 0 0.5rem 0;">Predictions</h3>
            <p style="color:#555; font-size:0.9rem;">
                Classifiez un produit en temps reel via l'API de prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# API health quick check
st.markdown("### Statut de l'API")
try:
    from utils import check_api_health

    health = check_api_health()
    if health.get("status") == "ok":
        st.success(
            f"API connectee - Modele **{health.get('model_name', 'N/A')}** charge avec succes."
        )
    else:
        st.warning(
            f"API en mode degrade : {health.get('detail', 'Pas de detail disponible.')}"
        )
except Exception:
    st.info(
        "Impossible de joindre l'API pour le moment. "
        "Verifiez que le service FastAPI est demarre."
    )
