import click
import pandas as pd
import re
import html
import os
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

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

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    os.makedirs(output_filepath, exist_ok=True)
    
    # on charge les fichiers sources
    raw_path = Path(input_filepath)
    transl_path = raw_path / "Rak_train_translations.csv"
    
    y = pd.read_csv(raw_path / "Y_train_CVw08PX.csv", index_col=0)
    
    if transl_path.exists():
        df = pd.read_csv(transl_path, index_col=0)
        df["product_txt"] = df["product_txt_transl"].fillna("").astype(str)
    else:
        x_train = pd.read_csv(raw_path / "X_train_update.csv", index_col=0)
        df = pd.DataFrame(index=x_train.index)
        df["product_txt"] = build_product_txt(x_train)

    df["prdtypecode"] = y.reindex(df.index)
    
    # split train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["prdtypecode"])
    
    train_df.to_csv(os.path.join(output_filepath, "train_clean.csv"))
    val_df.to_csv(os.path.join(output_filepath, "val_clean.csv"))
    print("✅ Ticket 12: splits generes")

if __name__ == "__main__":
    main()