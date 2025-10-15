from __future__ import annotations

"""Utility script to download and preprocess the Dow 30 dataset."""

import argparse
from pathlib import Path

import pandas as pd

from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", default="2009-01-01")
    parser.add_argument("--train-end", default="2020-07-01")
    parser.add_argument("--test-end", default="2021-12-31")
    parser.add_argument("--tickers", nargs="*", default=DOW_30_TICKER)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--train-output", default=None, help="Filename for the train split")
    parser.add_argument("--test-output", default=None, help="Filename for the test split")
    parser.add_argument("--include-vix", action="store_true")
    parser.add_argument("--include-turbulence", action="store_true")
    parser.add_argument(
        "--source-csv",
        type=str,
        help="Use an existing preprocessed dataset instead of downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_csv:
        processed_df = pd.read_csv(args.source_csv)
    else:
        downloader = YahooDownloader(
            start_date=args.train_start,
            end_date=args.test_end,
            ticker_list=args.tickers,
        )
        raw_df = downloader.fetch_data()

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=args.include_vix,
            use_turbulence=args.include_turbulence,
        )
        processed_df = fe.preprocess_data(raw_df)
        processed_df = processed_df.dropna().reset_index(drop=True)

    train_df = data_split(processed_df, args.train_start, args.train_end)
    test_df = data_split(processed_df, args.train_end, args.test_end)

    train_filename = args.train_output or "train_data.csv"
    test_filename = args.test_output or "test_data.csv"
    train_path = output_dir / train_filename
    test_path = output_dir / test_filename
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    print(f"Saved training data to {train_path}")
    print(f"Saved test data to {test_path}")


if __name__ == "__main__":
    main()
