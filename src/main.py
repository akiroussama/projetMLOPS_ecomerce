from data.make_dataset import make_dataset
from features.build_features import build_features
from models.train_model import train_model


# Training pipeline
def train_pipeline(preprocessed_path = "data/preprocessed"):
    # Pull data
    print('Pulling Data...')
    make_dataset(input_filepath='data/raw', output_filepath='data/preprocessed')

    # Build features
    print('Building Features...')
    X_train, X_val, y_train, y_val, feats = build_features(preprocessed_path = preprocessed_path)

    # Train model
    print('Training Model...')
    train_model(X_train, X_val, y_train, y_val, feats)


if __name__ == "__main__":
    train_pipeline()
