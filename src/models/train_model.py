import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

def main():
    root = Path("data/preprocessed")
    train = pd.read_csv(root / "train_clean.csv", index_col=0)
    
    x_train = train["product_txt"].fillna("").astype(str)
    y_train = train["prdtypecode"].astype(int)

    # construction du pipeline mika
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=120000, dtype=np.float32)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=160000, dtype=np.float32)
    
    feats = FeatureUnion([("word", word_vec), ("char", char_vec)])
    model = Pipeline([("feats", feats), ("clf", LinearSVC(C=0.5))])

    print("🏋️ Entrainement du modele final...")
    model.fit(x_train, y_train)
    
    os.makedirs("models/artifacts", exist_ok=True)
    joblib.dump(model, "models/artifacts/model_final.joblib")
    print("✅ Ticket 14: modele sauvegarde")

if __name__ == "__main__":
    main()