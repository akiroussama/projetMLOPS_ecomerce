import os
import pandas as pd
import numpy as np
import logging
import pickle
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# je telecharge ressources nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class TextFeatureEngineer:
    def __init__(self, max_features=5000):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))
        # on init tfidf
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        # je retire html
        text = BeautifulSoup(text, "html.parser").get_text()
        # je garde que lettres
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = word_tokenize(text.lower())
        # on filtre stop words
        filtered_words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(filtered_words)

    def prepare_df(self, df):
        # nous combinons titre et desc
        df['text_full'] = df['designation'].fillna('') + " " + df['description'].fillna('')
        df['text_clean'] = df['text_full'].apply(self.clean_text)
        return df

    def fit_transform(self, df):
        df_prep = self.prepare_df(df)
        # j'entraine et transforme le train
        X_tf = self.vectorizer.fit_transform(df_prep['text_clean'])
        return X_tf

    def transform(self, df):
        df_prep = self.prepare_df(df)
        # je transforme uniqment (val ou test)
        X_tf = self.vectorizer.transform(df_prep['text_clean'])
        return X_tf
        
    def save_vectorizer(self, path):
        # on save le modele pour api
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    
    logger.info('debut creation features texte')
    
    input_dir = 'data/preprocessed'
    output_dir = 'data/preprocessed'
    model_dir = 'models'
    
    # je lis donnees de la mission 3
    df_train = pd.read_csv(os.path.join(input_dir, 'X_train_clean.csv'))
    df_val = pd.read_csv(os.path.join(input_dir, 'X_val_clean.csv'))
    df_test = pd.read_csv(os.path.join(input_dir, 'X_test_clean.csv'))
    
    engineer = TextFeatureEngineer()
    
    # on applique transformation math
    logger.info('fit transform sur train')
    X_train_tf = engineer.fit_transform(df_train)
    
    logger.info('transform sur val et test')
    X_val_tf = engineer.transform(df_val)
    X_test_tf = engineer.transform(df_test)
    
    # on sauvegarde les matrices numpy
    import scipy.sparse
    scipy.sparse.save_npz(os.path.join(output_dir, 'X_train_tf.npz'), X_train_tf)
    scipy.sparse.save_npz(os.path.join(output_dir, 'X_val_tf.npz'), X_val_tf)
    scipy.sparse.save_npz(os.path.join(output_dir, 'X_test_tf.npz'), X_test_tf)
    
    # je sauvegarde mon pipeline pour l'api
    engineer.save_vectorizer(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    
    logger.info('features sauvegardees avec succes')