from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "prdtypecode"
TEXT_COLUMNS = ("designation", "description")
IDENTIFIER_COLUMNS = ("Unnamed: 0", "productid", "imageid")


def _normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in TEXT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    return cleaned


def _merge_training_frames(
    features_frame: pd.DataFrame, target_frame: pd.DataFrame
) -> pd.DataFrame:
    if TARGET_COLUMN not in target_frame.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in labels CSV.")

    for column in IDENTIFIER_COLUMNS:
        if column in features_frame.columns and column in target_frame.columns:
            labels = target_frame[[column, TARGET_COLUMN]]
            return features_frame.merge(labels, on=column, how="inner")

    merged = features_frame.copy()
    merged[TARGET_COLUMN] = target_frame[TARGET_COLUMN].values
    return merged


def prepare_datasets(
    input_filepath: str | Path,
    output_filepath: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, int]:
    """Clean the raw CSV files and create train/validation splits."""

    logger = logging.getLogger(__name__)
    input_dir = Path(input_filepath)
    output_dir = Path(output_filepath)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("prepare clean dataset from %s", input_dir)

    x_train_frame = pd.read_csv(input_dir / "X_train_update.csv")
    y_train_frame = pd.read_csv(input_dir / "Y_train_CVw08PX.csv")
    x_test_frame = pd.read_csv(input_dir / "X_test_update.csv")

    x_train_frame = _normalize_text_columns(x_train_frame)
    x_test_frame = _normalize_text_columns(x_test_frame)

    train_frame = _merge_training_frames(x_train_frame, y_train_frame)
    before_dedup = len(train_frame)
    train_frame = train_frame.drop_duplicates(
        subset=[
            column
            for column in ("designation", "description", "productid", "imageid")
            if column in train_frame.columns
        ]
    )
    duplicates_removed = before_dedup - len(train_frame)

    features_frame = train_frame.drop(columns=[TARGET_COLUMN])
    target_series = train_frame[TARGET_COLUMN]

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        features_frame,
        target_series,
        test_size=test_size,
        random_state=random_state,
        stratify=target_series,
    )

    x_train_split.to_csv(output_dir / "X_train_clean.csv", index=False)
    x_val_split.to_csv(output_dir / "X_val_clean.csv", index=False)
    x_test_frame.to_csv(output_dir / "X_test_clean.csv", index=False)

    pd.DataFrame({TARGET_COLUMN: y_train_split}).to_csv(
        output_dir / "Y_train_clean.csv", index=False
    )
    pd.DataFrame({TARGET_COLUMN: y_val_split}).to_csv(
        output_dir / "Y_val_clean.csv", index=False
    )

    return {
        "train_rows": int(len(x_train_split)),
        "validation_rows": int(len(x_val_split)),
        "test_rows": int(len(x_test_frame)),
        "duplicates_removed": int(duplicates_removed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean the raw Rakuten CSV files and create a validation split."
    )
    parser.add_argument("input_filepath", help="Directory containing the raw CSV files.")
    parser.add_argument(
        "output_filepath", help="Directory where the clean CSV files will be written."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the stratified split.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_parser().parse_args(argv)
    summary = prepare_datasets(
        args.input_filepath,
        args.output_filepath,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    logging.getLogger(__name__).info("clean dataset ready: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
