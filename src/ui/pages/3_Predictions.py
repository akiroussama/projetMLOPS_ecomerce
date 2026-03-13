"""
Page 3 - Predictions Live
==========================
Interactive prediction form that calls the FastAPI backend.
Includes quick-fill examples, confidence gauge, and session history.
"""

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Predictions | Rakuten MLOps",
    page_icon="R",
    layout="wide",
)

# Ensure utils is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import predict, check_api_health  # noqa: E402

# ---------------------------------------------------------------------------
# Category label mapping
# ---------------------------------------------------------------------------
CATEGORY_LABELS = {
    "10": "Livres / Magazines",
    "40": "Jeux video / Consoles",
    "50": "Accessoires gaming",
    "60": "Consoles de jeu",
    "1140": "Figurines / Pop culture",
    "1160": "Cartes collectionnables",
    "1180": "Jeux de societe / Figurines",
    "1280": "Jouets enfants",
    "1281": "Jeux d'eveil",
    "1300": "Modeles reduits / Maquettes",
    "1301": "Pieces detachees / Puericulture",
    "1302": "Jeux d'exterieur",
    "1320": "Puericulture / Bebe",
    "1560": "Mobilier interieur",
    "1920": "Literie / Oreillers",
    "1940": "Alimentation / Epicerie",
    "2060": "Decoration interieure",
    "2220": "Animaux de compagnie",
    "2280": "Magazines / Journaux",
    "2403": "Livres anciens / Collection",
    "2462": "Jeux / Consoles retro",
    "2522": "Papeterie / Fournitures",
    "2582": "Mobilier exterieur / Jardin",
    "2583": "Piscines / Accessoires",
    "2585": "Outillage / Bricolage",
    "2705": "Livres neufs",
    "2905": "Jeux PC / Software",
}

# ---------------------------------------------------------------------------
# Quick-fill examples
# ---------------------------------------------------------------------------
QUICK_EXAMPLES = [
    {
        "name": "Livre",
        "designation": "Le Petit Prince - Antoine de Saint-Exupery",
        "description": (
            "Magnifique edition collector du classique de la "
            "litterature francaise. Couverture rigide."
        ),
    },
    {
        "name": "Jeu PS5",
        "designation": "FIFA 24 Edition Standard PS5",
        "description": (
            "Le jeu de football ultime sur PlayStation 5. "
            "HyperMotion V, mode carriere renove."
        ),
    },
    {
        "name": "Figurine",
        "designation": "Figurine Pop! Marvel Spider-Man No Way Home",
        "description": (
            "Figurine Funko Pop en vinyle de Spider-Man. "
            "Taille environ 10cm. Emballage d'origine."
        ),
    },
    {
        "name": "Canape",
        "designation": "Canape d'angle convertible tissu gris 4 places",
        "description": (
            "Grand canape d'angle avec fonction couchage. "
            "Tissu gris anthracite, coussins dehoussables."
        ),
    },
    {
        "name": "Perceuse",
        "designation": "Perceuse visseuse sans fil Bosch Pro 18V",
        "description": (
            "Perceuse visseuse Bosch GSR 18V-60 avec 2 batteries "
            "4.0Ah, chargeur rapide et coffret L-BOXX."
        ),
    },
    {
        "name": "Carte Pokemon",
        "designation": "Display 36 boosters Pokemon EV3.5",
        "description": (
            "Boite complete de 36 boosters Pokemon extension "
            "Ecarlate et Violet 151. Version francaise."
        ),
    },
]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "first_prediction" not in st.session_state:
    st.session_state.first_prediction = True

if "quick_fill_desig" not in st.session_state:
    st.session_state.quick_fill_desig = ""

if "quick_fill_desc" not in st.session_state:
    st.session_state.quick_fill_desc = ""


# ---------------------------------------------------------------------------
# Confidence gauge
# ---------------------------------------------------------------------------
def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a plotly gauge chart for prediction confidence."""
    if confidence is None:
        confidence = 0.0

    pct = confidence * 100

    # Colour based on confidence level
    if pct >= 70:
        bar_color = "#2ECC40"
        text_label = "Haute confiance"
    elif pct >= 40:
        bar_color = "#FF851B"
        text_label = "Confiance moyenne"
    else:
        bar_color = "#FF4136"
        text_label = "Faible confiance"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 48, "color": "#1A1A2E"}},
            title={"text": text_label, "font": {"size": 16, "color": "#555"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E0E0E0"},
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "#F5F5F5",
                "borderwidth": 2,
                "bordercolor": "#E0E0E0",
                "steps": [
                    {"range": [0, 40], "color": "#FFE5E5"},
                    {"range": [40, 70], "color": "#FFF3E0"},
                    {"range": [70, 100], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line": {"color": "#BF0000", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">
            🔮 Predictions en Temps Reel
        </h1>
        <p style="font-size: 1.05rem; color: #555;">
            Classifiez un produit instantanement via l'API de prediction
        </p>
    </div>
    <hr style="border:none; height:2px; background:linear-gradient(90deg,#BF0000,transparent); margin-bottom:1.5rem;">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API status indicator
# ---------------------------------------------------------------------------
api_ok = False
try:
    health = check_api_health()
    if health.get("status") == "ok":
        api_ok = True
        st.markdown(
            '<div style="background:#E8F5E9; border-left:4px solid #2ECC40; '
            'padding:0.6rem 1rem; border-radius:6px; margin-bottom:1rem;">'
            '✅ <b>API connectee</b> - Modele <code>{}</code> pret pour les predictions.'
            '</div>'.format(health.get("model_name", "N/A")),
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            f"API en mode degrade : {health.get('detail', 'Pas de detail')}. "
            "Les predictions peuvent echouer."
        )
except Exception:
    st.markdown(
        '<div style="background:#FFF3E0; border-left:4px solid #FF851B; '
        'padding:0.6rem 1rem; border-radius:6px; margin-bottom:1rem;">'
        '⚠️ <b>API non joignable</b> - Verifiez que le service FastAPI est demarre. '
        'Les predictions ne sont pas disponibles pour le moment.'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Quick-fill example buttons
# ---------------------------------------------------------------------------
st.markdown("### Exemples rapides")
st.caption("Cliquez sur un exemple pour pre-remplir le formulaire :")

cols = st.columns(len(QUICK_EXAMPLES))
for idx, example in enumerate(QUICK_EXAMPLES):
    with cols[idx]:
        if st.button(
            f"📦 {example['name']}",
            key=f"example_{idx}",
            use_container_width=True,
        ):
            st.session_state.quick_fill_desig = example["designation"]
            st.session_state.quick_fill_desc = example["description"]
            st.rerun()

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:1rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction form
# ---------------------------------------------------------------------------
st.markdown("### Formulaire de Prediction")

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    with st.form("prediction_form", clear_on_submit=False):
        designation = st.text_input(
            "Designation du produit *",
            value=st.session_state.quick_fill_desig,
            placeholder="Ex: Figurine Pop! Spider-Man No Way Home",
            help="Titre court du produit (obligatoire, max 512 caracteres)",
        )

        description = st.text_area(
            "Description du produit",
            value=st.session_state.quick_fill_desc,
            placeholder="Ex: Figurine en vinyle de Spider-Man, taille 10cm, emballage d'origine...",
            help="Description longue du produit (optionnel)",
            height=150,
        )

        submitted = st.form_submit_button(
            "🔮 Predire la categorie",
            use_container_width=True,
            type="primary",
        )

    # Clear quick-fill after form rendering
    if st.session_state.quick_fill_desig or st.session_state.quick_fill_desc:
        st.session_state.quick_fill_desig = ""
        st.session_state.quick_fill_desc = ""

with col_result:
    if submitted:
        if not designation or not designation.strip():
            st.error("La designation est obligatoire. Veuillez saisir au moins un mot.")
        else:
            with st.spinner("Classification en cours..."):
                try:
                    start_time = time.time()
                    result = predict(designation.strip(), description.strip() if description else "")
                    elapsed = time.time() - start_time

                    predicted_code = str(result.get("predicted_code", ""))
                    predicted_label = result.get("predicted_label", predicted_code)
                    confidence = result.get("confidence")
                    model_name = result.get("model_name", "N/A")

                    # Map category label
                    category_name = CATEGORY_LABELS.get(predicted_code, predicted_label)

                    # Success display
                    st.markdown(
                        """
                        <div style="background: linear-gradient(135deg, #BF0000, #8B0000);
                                    border-radius: 12px; padding: 1.5rem; color: white;
                                    text-align: center; margin-bottom: 1rem;">
                            <p style="margin:0; font-size:0.85rem; opacity:0.8; text-transform:uppercase;
                                      letter-spacing:1px;">Categorie Predite</p>
                            <h2 style="margin: 0.3rem 0; font-size: 1.8rem; color: white !important;">
                                {category_name}
                            </h2>
                            <p style="margin:0; font-size: 1rem;">
                                <span style="background:rgba(255,255,255,0.2); padding:3px 12px;
                                             border-radius:15px;">Code: {predicted_code}</span>
                            </p>
                        </div>
                        """.format(category_name=category_name, predicted_code=predicted_code),
                        unsafe_allow_html=True,
                    )

                    # Confidence gauge
                    if confidence is not None:
                        fig = create_confidence_gauge(confidence)
                        st.plotly_chart(fig, use_container_width=True)

                    # Extra info
                    info_cols = st.columns(2)
                    with info_cols[0]:
                        st.metric("Modele", model_name)
                    with info_cols[1]:
                        st.metric("Temps de reponse", f"{elapsed:.2f}s")

                    # Balloons on first prediction
                    if st.session_state.first_prediction:
                        st.balloons()
                        st.session_state.first_prediction = False

                    # Add to session history
                    st.session_state.prediction_history.append({
                        "Designation": designation[:80],
                        "Code": predicted_code,
                        "Categorie": category_name,
                        "Confiance": f"{confidence * 100:.1f}%" if confidence else "N/A",
                        "Modele": model_name,
                        "Temps": f"{elapsed:.2f}s",
                    })

                except Exception as e:
                    error_msg = str(e)
                    if "ConnectionError" in type(e).__name__ or "Connection" in error_msg:
                        st.error(
                            "Impossible de joindre l'API de prediction. "
                            "Verifiez que le service FastAPI est demarre et accessible."
                        )
                    elif "401" in error_msg or "403" in error_msg:
                        st.error(
                            "Erreur d'authentification. Verifiez que le token "
                            "API_AUTH_TOKEN est correctement configure."
                        )
                    elif "422" in error_msg:
                        st.error(
                            "Donnees invalides. Verifiez que la designation "
                            "n'est pas vide et ne depasse pas 512 caracteres."
                        )
                    elif "503" in error_msg:
                        st.error(
                            "Le modele n'est pas encore charge. "
                            "Veuillez patienter quelques instants et reessayer."
                        )
                    else:
                        st.error(f"Erreur lors de la prediction : {error_msg}")
    else:
        # Placeholder when no prediction yet
        st.markdown(
            """
            <div style="background:#F8F9FA; border:2px dashed #E0E0E0;
                        border-radius:12px; padding:3rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:0.5rem; opacity:0.5;">🔮</div>
                <p style="color:#999; font-size:1.05rem; margin:0;">
                    Remplissez le formulaire et cliquez sur
                    <b>"Predire la categorie"</b> pour voir le resultat ici.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:2rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session prediction history
# ---------------------------------------------------------------------------
st.markdown("### Historique des Predictions (session)")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    # Reverse so newest appears first
    history_df = history_df.iloc[::-1].reset_index(drop=True)
    history_df.index = history_df.index + 1
    history_df.index.name = "#"

    st.dataframe(history_df, use_container_width=True)

    col_clear, col_count = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    with col_count:
        st.caption(f"{len(st.session_state.prediction_history)} prediction(s) dans cette session.")
else:
    st.markdown(
        """
        <div style="background:#F8F9FA; border-radius:8px; padding:1.5rem;
                    text-align:center; border:1px solid #E0E0E0;">
            <p style="color:#999; margin:0;">
                Aucune prediction pour le moment. Lancez votre premiere prediction ci-dessus !
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")
st.caption("Les predictions sont envoyees a l'API FastAPI via HTTP. Aucun modele n'est charge directement.")
