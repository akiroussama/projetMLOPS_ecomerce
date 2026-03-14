'''
Make Dataset

Example usage:
python3 src/data/make_dataset.py data/raw data/preprocessed
'''

import pandas as pd
import re
import html
import os
from pathlib import Path
from bs4 import BeautifulSoup

def strip_html(text):
    text = "" if text is None else str(text)
    text = html.unescape(text)
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def basic_clean(text):
    text = "" if text is None else str(text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def build_product_txt(df):
    des = df.get("designation", pd.Series([""] * len(df))).fillna("").astype(str)
    desc = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)
    des = des.map(strip_html).map(basic_clean)
    desc = desc.map(strip_html).map(basic_clean)
    return (des + " . -//- " + desc).fillna("").astype(str)

def make_dataset(input_filepath='data/raw', output_filepath='data/preprocessed'):
    os.makedirs(output_filepath, exist_ok=True)
    
    # on charge les fichiers sources
    raw_path = Path(input_filepath)
    preprocessed_path = Path(output_filepath)
    print("raw_path:", raw_path)
    print("preprocessed_path:", preprocessed_path)

    # Fichiers bruts
    X_train_path = raw_path / "X_train_update.csv"
    y_train_path = raw_path / "Y_train_CVw08PX.csv"

    # Fichier déjà traduit
    transl_path = raw_path / "Rak_train_translations.csv"
    
    # Labels
    y = pd.read_csv(y_train_path, index_col=0).iloc[:, 0]
    
    if transl_path.exists():
        print(f"Using translations: {transl_path}")
        df = pd.read_csv(transl_path, index_col=0)
        if "product_txt_transl" not in df.columns:
            raise ValueError("Rak_train_translations.csv doit contenir la colonne 'product_txt_transl'")
    else:
        print("No translations file found. Building product_txt from raw.")
        df = pd.read_csv(X_train_path, index_col=0)
        df["product_txt"] = build_product_txt(df)

    #on rajoute le label
    df["prdtypecode"] = y.reindex(df.index)
    df["prdtypecode"] = df["prdtypecode"].astype(int)
    missing_pdtcd = df["prdtypecode"].isna().sum()
    if missing_pdtcd > 0:
        raise ValueError(f"Error: {missing_pdtcd} labels manquants après alignement (index mismatch).")

    # Nettoyage final anti-NaN
    df["product_txt"] = df["product_txt"].fillna("").astype(str)
    
    # Clean CSV path
    out_path = preprocessed_path / "train_clean.csv"
    df[["product_txt", "prdtypecode"]].to_csv(out_path)
    print(f"Dataset: {len(df)} lignes")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    make_dataset()