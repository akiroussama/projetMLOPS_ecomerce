import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_features(preprocessed_path = "data/preprocessed"):
    ##load data prepared by make_dataset
    print("Using preprocessed_path:", preprocessed_path)
    path_train_clean = f"{preprocessed_path}/train_clean.csv"

    if not os.path.exists(path_train_clean):
        raise FileNotFoundError(f"Fichier introuvable: {path_train_clean}")
    
    df = pd.read_csv(path_train_clean, index_col=0)

    X = df["product_txt"]
    y = df["prdtypecode"]


    # =========================
    # Split once
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for i in (X_train, X_val, y_train, y_val): 
        print(i.shape)


    # =========================
    # Vectorize ONCE (best stable defaults)
    # =========================
    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=120_000,
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=160_000,
    )

    feats = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])

    print("Vectorization done")

    return (X_train, X_val, y_train, y_val, feats)


if __name__ == "__main__":
    build_features()