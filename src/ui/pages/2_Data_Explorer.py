"""
Page 2 - Data Explorer
=======================
Explore training data: category distribution, product examples, descriptive stats.
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Data Explorer | Rakuten MLOps",
    page_icon="\ud83d\uded2",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Category label mapping (prdtypecode -> readable name)
# ---------------------------------------------------------------------------
CATEGORY_LABELS = {
    10: "Livres / Magazines",
    40: "Jeux video / Consoles",
    50: "Accessoires gaming",
    60: "Consoles de jeu",
    1140: "Figurines / Pop culture",
    1160: "Cartes collectionnables",
    1180: "Jeux de societe / Figurines",
    1280: "Jouets enfants",
    1281: "Jeux d'eveil",
    1300: "Modeles reduits / Maquettes",
    1301: "Pieces detachees / Puericulture",
    1302: "Jeux d'exterieur",
    1320: "Puericulture / Bebe",
    1560: "Mobilier interieur",
    1920: "Literie / Oreillers",
    1940: "Alimentation / Epicerie",
    2060: "Decoration interieure",
    2220: "Animaux de compagnie",
    2280: "Magazines / Journaux",
    2403: "Livres anciens / Collection",
    2462: "Jeux / Consoles retro",
    2522: "Papeterie / Fournitures",
    2582: "Mobilier exterieur / Jardin",
    2583: "Piscines / Accessoires",
    2585: "Outillage / Bricolage",
    2705: "Livres neufs",
    2905: "Jeux PC / Software",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Try to load the preprocessed CSV from the mounted volume.
    Falls back to an embedded sample if the file is not found.
    """
    candidate_paths = [
        "data/preprocessed/X_train_clean.csv",
        "/app/data/preprocessed/X_train_clean.csv",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "preprocessed", "X_train_clean.csv"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8")
                return df
            except Exception:
                try:
                    df = pd.read_csv(path, encoding="latin-1")
                    return df
                except Exception:
                    continue

    # Fallback: generate a small demo dataset
    return _generate_demo_data()


def _generate_demo_data() -> pd.DataFrame:
    """Generate a realistic demo dataset when no CSV is available."""
    import random
    random.seed(42)

    categories = list(CATEGORY_LABELS.keys())
    sample_designations = {
        10: [
            "Le Petit Prince - Antoine de Saint-Exupery",
            "Harry Potter et la Chambre des Secrets",
            "L'Alchimiste - Paulo Coelho",
            "Les Miserables - Victor Hugo",
        ],
        40: [
            "FIFA 24 - PS5",
            "The Legend of Zelda: Tears of the Kingdom",
            "Grand Theft Auto V - Xbox Series X",
            "Minecraft - Nintendo Switch",
        ],
        50: [
            "Manette PS5 DualSense Blanche",
            "Casque gaming HyperX Cloud II",
            "Clavier mecanique RGB Corsair K70",
            "Souris gaming Logitech G502",
        ],
        60: [
            "PlayStation 5 Console Standard",
            "Nintendo Switch OLED Blanche",
            "Xbox Series X 1 To",
            "Steam Deck 256 Go",
        ],
        1140: [
            "Figurine Pop! Spider-Man No Way Home",
            "Figurine Dragon Ball Z - Goku Ultra Instinct",
            "Statue One Piece - Luffy Gear 5",
            "Figurine Naruto Shippuden - Sasuke",
        ],
        1160: [
            "Booster Pokemon Ecarlate et Violet",
            "Carte Yu-Gi-Oh! Collection Legendaire",
            "Display Pokemon 151",
            "Starter Deck One Piece Card Game",
        ],
        1560: [
            "Canape d'angle convertible gris",
            "Table basse scandinave en bois",
            "Etagere murale design 3 niveaux",
            "Bureau d'angle avec rangements",
        ],
        2060: [
            "Tableau toile abstraite 80x120cm",
            "Miroir rond dore 60cm",
            "Horloge murale vintage industrielle",
            "Vase en ceramique fait main",
        ],
        2522: [
            "Lot de 100 stylos Bic Cristal",
            "Agenda 2025 Moleskine noir",
            "Cahier Oxford A4 grands carreaux",
            "Trousse en cuir vintage",
        ],
        2585: [
            "Perceuse visseuse Bosch 18V",
            "Coffret d'outils 150 pieces",
            "Scie circulaire Makita 1400W",
            "Niveau laser Bosch GLL 3-80",
        ],
    }

    rows = []
    for _ in range(500):
        cat = random.choice(categories)
        desigs = sample_designations.get(cat, [f"Produit categorie {cat} - Article #{random.randint(1000, 9999)}"])
        desig = random.choice(desigs)
        desc = f"Description detaillee du produit. Categorie {cat}. " * random.randint(1, 4)
        rows.append({
            "designation": desig,
            "description": desc,
            "prdtypecode": cat,
            "productid": random.randint(100000, 999999),
            "imageid": random.randint(1000000, 9999999),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">
            📊 Data Explorer
        </h1>
        <p style="font-size: 1.05rem; color: #555;">
            Exploration du dataset Rakuten - Analyse des donnees d'entrainement
        </p>
    </div>
    <hr style="border:none; height:2px; background:linear-gradient(90deg,#BF0000,transparent); margin-bottom:1.5rem;">
    """,
    unsafe_allow_html=True,
)

# Load data
with st.spinner("Chargement des donnees..."):
    df = load_data()

# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
st.markdown("## Vue d'ensemble")

has_target = "prdtypecode" in df.columns
has_desig = "designation" in df.columns
has_desc = "description" in df.columns

# Compute stats
n_products = len(df)
n_categories = df["prdtypecode"].nunique() if has_target else "N/A"
avg_desig_len = int(df["designation"].astype(str).str.len().mean()) if has_desig else "N/A"
avg_desc_len = int(df["description"].astype(str).str.len().mean()) if has_desc else "N/A"

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Nombre de produits", f"{n_products:,}")
with m2:
    st.metric("Categories uniques", n_categories)
with m3:
    st.metric("Longueur moy. designation", f"{avg_desig_len} car." if isinstance(avg_desig_len, int) else avg_desig_len)
with m4:
    st.metric("Longueur moy. description", f"{avg_desc_len} car." if isinstance(avg_desc_len, int) else avg_desc_len)

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:1.5rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Category distribution
# ---------------------------------------------------------------------------
if has_target:
    st.markdown("## Distribution des Categories")

    cat_counts = df["prdtypecode"].value_counts().reset_index()
    cat_counts.columns = ["prdtypecode", "count"]
    cat_counts["label"] = cat_counts["prdtypecode"].map(
        lambda x: CATEGORY_LABELS.get(x, f"Code {x}")
    )
    cat_counts["display"] = cat_counts.apply(
        lambda r: f"{r['prdtypecode']} - {r['label']}", axis=1
    )
    cat_counts = cat_counts.sort_values("count", ascending=True)

    tab_bar, tab_pie = st.tabs(["Bar Chart", "Pie Chart"])

    with tab_bar:
        fig_bar = px.bar(
            cat_counts,
            x="count",
            y="display",
            orientation="h",
            color="count",
            color_continuous_scale=["#FFE5E5", "#BF0000"],
            labels={"count": "Nombre de produits", "display": "Categorie"},
        )
        fig_bar.update_layout(
            title="Nombre de produits par categorie",
            height=max(500, len(cat_counts) * 28),
            yaxis_title="",
            xaxis_title="Nombre de produits",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_pie:
        fig_pie = px.pie(
            cat_counts,
            values="count",
            names="display",
            color_discrete_sequence=px.colors.sequential.Reds_r,
        )
        fig_pie.update_layout(
            title="Repartition des categories",
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:1.5rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Text length distribution
# ---------------------------------------------------------------------------
if has_desig:
    st.markdown("## Analyse des Longueurs de Texte")

    col_hist1, col_hist2 = st.columns(2)

    with col_hist1:
        desig_lengths = df["designation"].astype(str).str.len()
        fig_desig = px.histogram(
            desig_lengths,
            nbins=50,
            color_discrete_sequence=["#BF0000"],
            labels={"value": "Longueur (caracteres)", "count": "Frequence"},
        )
        fig_desig.update_layout(
            title="Distribution de la longueur des designations",
            xaxis_title="Longueur (caracteres)",
            yaxis_title="Frequence",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig_desig, use_container_width=True)

    with col_hist2:
        if has_desc:
            desc_lengths = df["description"].astype(str).str.len()
            fig_desc = px.histogram(
                desc_lengths,
                nbins=50,
                color_discrete_sequence=["#8B0000"],
                labels={"value": "Longueur (caracteres)", "count": "Frequence"},
            )
            fig_desc.update_layout(
                title="Distribution de la longueur des descriptions",
                xaxis_title="Longueur (caracteres)",
                yaxis_title="Frequence",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_desc, use_container_width=True)

    st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:1.5rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Product examples (filterable table)
# ---------------------------------------------------------------------------
st.markdown("## Exemples de Produits")

if has_target:
    # Category filter
    all_categories = sorted(df["prdtypecode"].unique())
    category_options = {
        f"{code} - {CATEGORY_LABELS.get(code, 'N/A')}": code for code in all_categories
    }

    selected_label = st.selectbox(
        "Filtrer par categorie",
        options=["Toutes les categories"] + list(category_options.keys()),
    )

    if selected_label == "Toutes les categories":
        filtered_df = df
    else:
        selected_code = category_options[selected_label]
        filtered_df = df[df["prdtypecode"] == selected_code]

    # Number of rows to display
    n_display = st.slider("Nombre de produits a afficher", 5, 100, 20, 5)

    display_cols = [c for c in ["prdtypecode", "designation", "description"] if c in filtered_df.columns]
    display_df = filtered_df[display_cols].head(n_display).copy()

    # Truncate long descriptions for display
    if "description" in display_df.columns:
        display_df["description"] = display_df["description"].astype(str).str[:200] + "..."

    if "prdtypecode" in display_df.columns:
        display_df["categorie"] = display_df["prdtypecode"].map(
            lambda x: CATEGORY_LABELS.get(x, f"Code {x}")
        )
        display_df = display_df[["prdtypecode", "categorie"] + [c for c in display_cols if c != "prdtypecode"]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Affichage de {len(display_df)} sur {len(filtered_df)} produits filtres ({n_products} total).")
else:
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.markdown('<hr style="border:none; height:1px; background:#e0e0e0; margin:1.5rem 0;">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Descriptive stats table
# ---------------------------------------------------------------------------
st.markdown("## Statistiques Descriptives")

stats_data = {}

if has_desig:
    desig_lens = df["designation"].astype(str).str.len()
    stats_data["Designation"] = {
        "Min": int(desig_lens.min()),
        "Max": int(desig_lens.max()),
        "Moyenne": round(desig_lens.mean(), 1),
        "Mediane": int(desig_lens.median()),
        "Ecart-type": round(desig_lens.std(), 1),
    }

if has_desc:
    desc_lens = df["description"].astype(str).str.len()
    stats_data["Description"] = {
        "Min": int(desc_lens.min()),
        "Max": int(desc_lens.max()),
        "Moyenne": round(desc_lens.mean(), 1),
        "Mediane": int(desc_lens.median()),
        "Ecart-type": round(desc_lens.std(), 1),
    }

if stats_data:
    stats_df = pd.DataFrame(stats_data).T
    stats_df.index.name = "Champ"
    st.dataframe(stats_df, use_container_width=True)

st.markdown("")
st.caption("Donnees chargees depuis le volume data/preprocessed/ ou le jeu de demo integre.")
