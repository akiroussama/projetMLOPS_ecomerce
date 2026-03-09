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
from sklearn.model_selection import train_test_split

# je telecharge ressources nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class DataImporter:
    def load_data(self):
        # Load raw data
        x_train_path = 'data/raw/X_train_update.csv'
        y_train_path = 'data/raw/Y_train_CVw08PX.csv'
        x_test_path = 'data/raw/X_test_update.csv'

        df_x_train = pd.read_csv(x_train_path)
        df_y_train = pd.read_csv(y_train_path)
        df_x_test = pd.read_csv(x_test_path)

        # Clean NA
        df_x_train['description'] = df_x_train['description'].fillna('')
        df_x_test['description'] = df_x_test['description'].fillna('')

        # Merge x and y for train
        df_train_full = pd.merge(df_x_train, df_y_train, on='Unnamed: 0')
        df_train_full = df_train_full.drop_duplicates(subset=['designation', 'description', 'productid', 'imageid'])

        # Add image_path
        df_train_full['image_path'] = df_train_full.apply(lambda row: f"data/raw/image_train/image_{row['imageid']}_product_{row['productid']}.jpg", axis=1)

        return df_train_full, df_x_test

    def split_train_test(self, df_train_full):
        y_col = 'prdtypecode'
        df_y_clean = df_train_full[['Unnamed: 0', y_col]]
        df_x_clean = df_train_full.drop(columns=[y_col])

        # Split train val stratified
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            df_x_clean, df_y_clean, test_size=0.2, random_state=42, stratify=df_y_clean[y_col]
        )

        return x_train_split, x_val_split, None, y_train_split[y_col], y_val_split[y_col], None

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        # Remove html
        text = BeautifulSoup(text, "html.parser").get_text()
        # Keep only letters
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = word_tokenize(text.lower())
        # Filter stop words
        filtered_words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(filtered_words)

    def preprocess_text_in_df(self, df, columns):
        for col in columns:
            df[col] = df[col].apply(self.clean_text)

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size

    def preprocess_images_in_df(self, df):
        from tensorflow.keras.applications.vgg16 import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        import numpy as np

        images = []
        for idx, row in df.iterrows():
            img_path = f"data/raw/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
            try:
                img = load_img(img_path, target_size=self.img_size)
                img_array = img_to_array(img)
                img_preprocessed = preprocess_input(img_array)
                images.append(img_preprocessed)
            except:
                # If image not found, use zeros
                images.append(np.zeros((224, 224, 3)))
        df['image_features'] = images

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