from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests


DEFAULT_FILENAMES = (
    "X_test_update.csv",
    "X_train_update.csv",
    "Y_train_CVw08PX.csv",
)
DEFAULT_BUCKET_URL = (
    "https://mlops-project-db.s3.eu-west-1.amazonaws.com/classification_e-commerce/"
)


def import_raw_data(
    raw_data_relative_path: str | Path = "data/raw",
    filenames: Iterable[str] = DEFAULT_FILENAMES,
    bucket_folder_url: str = DEFAULT_BUCKET_URL,
    *,
    skip_existing: bool = True,
    timeout: int = 60,
) -> dict[str, int]:
    """Download the tabular raw data used by the baseline training pipeline."""

    output_dir = Path(raw_data_relative_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    downloaded = 0
    skipped = 0

    with requests.Session() as session:
        for filename in filenames:
            destination = output_dir / filename

            if skip_existing and destination.exists():
                skipped += 1
                logger.info("skip existing raw file: %s", destination.name)
                continue

            file_url = urljoin(bucket_folder_url.rstrip("/") + "/", filename)
            logger.info("download raw file: %s", file_url)
            response = session.get(file_url, timeout=timeout)
            response.raise_for_status()
            destination.write_bytes(response.content)
            downloaded += 1

    return {"downloaded": downloaded, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the raw CSV files used by the Rakuten baseline pipeline."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where the raw CSV files will be stored.",
    )
    parser.add_argument(
        "--bucket-url",
        default=DEFAULT_BUCKET_URL,
        help="Base URL hosting the raw CSV files.",
    )
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="Filename to download. Repeat the flag to override the default file list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each file download.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = build_parser().parse_args(argv)
    summary = import_raw_data(
        raw_data_relative_path=args.output_dir,
        filenames=args.filenames or DEFAULT_FILENAMES,
        bucket_folder_url=args.bucket_url,
        skip_existing=not args.force,
        timeout=args.timeout,
    )
    logging.getLogger(__name__).info(
        "raw data ready: %s downloaded, %s skipped",
        summary["downloaded"],
        summary["skipped"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
