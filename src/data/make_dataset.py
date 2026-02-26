import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # on cree dossier preprocessed si besoin
    os.makedirs(output_filepath, exist_ok=True)

    # je charge donnees brutes
    x_train_path = os.path.join(input_filepath, 'X_train_update.csv')
    y_train_path = os.path.join(input_filepath, 'Y_train_CVw08PX.csv')
    x_test_path = os.path.join(input_filepath, 'X_test_update.csv')

    df_x_train = pd.read_csv(x_train_path)
    df_y_train = pd.read_csv(y_train_path)
    df_x_test = pd.read_csv(x_test_path)

    # on nettoie na en remettant texte vide
    df_x_train['description'] = df_x_train['description'].fillna('')
    df_x_test['description'] = df_x_test['description'].fillna('')

    # nous fusionnons x et y pr drop doublons
    df_train_full = pd.merge(df_x_train, df_y_train, on='Unnamed: 0')
    df_train_full = df_train_full.drop_duplicates(subset=['designation', 'description', 'productid', 'imageid'])

    # on separe cible
    y_col = 'prdtypecode'
    df_y_clean = df_train_full[['Unnamed: 0', y_col]]
    df_x_clean = df_train_full.drop(columns=[y_col])

    # je split train val stratifie
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        df_x_clean, df_y_clean, test_size=0.2, random_state=42, stratify=df_y_clean[y_col]
    )

    # on save csv propres
    x_train_split.to_csv(os.path.join(output_filepath, 'X_train_clean.csv'), index=False)
    y_train_split.to_csv(os.path.join(output_filepath, 'Y_train_clean.csv'), index=False)
    x_val_split.to_csv(os.path.join(output_filepath, 'X_val_clean.csv'), index=False)
    y_val_split.to_csv(os.path.join(output_filepath, 'Y_val_clean.csv'), index=False)
    df_x_test.to_csv(os.path.join(output_filepath, 'X_test_clean.csv'), index=False)

    # on copie images
    img_train_src = os.path.join(input_filepath, 'image_train')
    img_train_dst = os.path.join(output_filepath, 'image_train')
    if os.path.exists(img_train_src) and not os.path.exists(img_train_dst):
        shutil.copytree(img_train_src, img_train_dst)

if __name__ == '__main__':
    log_fmt = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()