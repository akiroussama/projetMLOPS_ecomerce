from __future__ import annotations

import argparse
import html
import logging
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str | None) -> str:
    if text is None or pd.isna(text):
        return ""

    raw_text = html.unescape(str(text))
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)
    raw_text = unicodedata.normalize("NFKD", raw_text)
    raw_text = raw_text.encode("ascii", "ignore").decode("ascii")
    raw_text = re.sub(r"[^a-zA-Z0-9]+", " ", raw_text.lower())
    return re.sub(r"\s+", " ", raw_text).strip()


@dataclass
class TextFeatureEngineer:
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(self.ngram_min, self.ngram_max),
            preprocessor=normalize_text,
        )

    @staticmethod
    def prepare_dataframe(dataframe: pd.DataFrame) -> pd.Series:
        designation = dataframe.get("designation", pd.Series("", index=dataframe.index))
        description = dataframe.get("description", pd.Series("", index=dataframe.index))
        return designation.fillna("").astype(str).str.strip() + " " + description.fillna(
            ""
        ).astype(str).str.strip()

    def fit_transform(self, dataframe: pd.DataFrame):
        return self.vectorizer.fit_transform(self.prepare_dataframe(dataframe))

    def transform(self, dataframe: pd.DataFrame):
        return self.vectorizer.transform(self.prepare_dataframe(dataframe))

    def save_vectorizer(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file_handle:
            pickle.dump(self.vectorizer, file_handle)
        return destination


def build_feature_matrices(
    *,
    input_dir: str | Path = "data/preprocessed",
    output_dir: str | Path = "data/preprocessed",
    model_dir: str | Path = "models",
    max_features: int = 5000,
) -> dict[str, int]:
    logger = logging.getLogger(__name__)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    model_path = Path(model_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(input_path / "X_train_clean.csv")
    df_val = pd.read_csv(input_path / "X_val_clean.csv")
    df_test = pd.read_csv(input_path / "X_test_clean.csv")

    engineer = TextFeatureEngineer(max_features=max_features)

    logger.info("build TF-IDF features")
    x_train_tf = engineer.fit_transform(df_train)
    x_val_tf = engineer.transform(df_val)
    x_test_tf = engineer.transform(df_test)

    scipy.sparse.save_npz(output_path / "X_train_tf.npz", x_train_tf)
    scipy.sparse.save_npz(output_path / "X_val_tf.npz", x_val_tf)
    scipy.sparse.save_npz(output_path / "X_test_tf.npz", x_test_tf)
    engineer.save_vectorizer(model_path / "tfidf_vectorizer.pkl")

    return {
        "train_rows": int(x_train_tf.shape[0]),
        "validation_rows": int(x_val_tf.shape[0]),
        "test_rows": int(x_test_tf.shape[0]),
        "feature_count": int(x_train_tf.shape[1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create TF-IDF features from the clean Rakuten CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/preprocessed",
        help="Directory containing X_train_clean.csv / X_val_clean.csv / X_test_clean.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/preprocessed",
        help="Directory where the sparse feature matrices will be written.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where the serialized TF-IDF vectorizer will be written.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features to keep.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_parser().parse_args(argv)
    summary = build_feature_matrices(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_features=args.max_features,
    )
    logging.getLogger(__name__).info("feature matrices ready: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
