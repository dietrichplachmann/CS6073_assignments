"""Prepare the numeric county features used by the HW1 baseline model."""

import argparse

import numpy as np
import pandas as pd


TARGET = "TARGET_deathRate"
DEFAULT_DROP_COLUMNS = ("id", "Geography", "binnedInc", "PctSomeCol18_24")


def strip_columns(frame, columns=DEFAULT_DROP_COLUMNS):
    """Return a copy without the listed columns; reject misspelled names."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not found: {sorted(missing)}")
    return frame.drop(columns=list(columns)).copy()


class NumericPreprocessor:
    """Learn median imputation and standardization from training rows only."""

    def __init__(self, drop_columns=DEFAULT_DROP_COLUMNS):
        self.drop_columns = tuple(drop_columns)

    def fit(self, frame):
        features = strip_columns(frame, self.drop_columns).drop(columns=[TARGET], errors="ignore")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise ValueError("All remaining features must be numeric")
        self.columns = list(features.columns)
        self.medians = features.median().fillna(0.0)
        filled = features.fillna(self.medians)
        self.means = filled.mean()
        self.scales = filled.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, frame):
        features = strip_columns(frame, self.drop_columns).drop(columns=[TARGET], errors="ignore")
        if list(features.columns) != self.columns:
            raise ValueError("Input feature columns differ from the training columns")
        values = (features.fillna(self.medians) - self.means) / self.scales
        return values.to_numpy(dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Remove selected columns from a CSV file")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument(
        "--drop-columns", nargs="+", default=DEFAULT_DROP_COLUMNS,
        help="Column names to remove (default: id, Geography, binnedInc, PctSomeCol18_24)",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    strip_columns(frame, args.drop_columns).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
