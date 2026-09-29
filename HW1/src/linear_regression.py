"""Train and evaluate a simple SGD linear regression baseline for HW1."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import NumericPreprocessor, TARGET


NUMERIC_DROP_COLUMNS = ("id", "Geography", "binnedInc", "PctSomeCol18_24")


def scores(actual, predicted):
    mse = float(np.mean((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((actual - predicted) ** 2)) / total
    return mse, r_squared


def train_sgd(x_train, y_train, x_val, y_val, epochs, learning_rate, seed):
    """Update a linear model after each randomly ordered training sample."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(x_train.shape[1] + 1, dtype=np.float64)
    train_design = np.column_stack((np.ones(len(x_train)), x_train))
    val_design = np.column_stack((np.ones(len(x_val)), x_val))
    history = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_weights = weights.copy()

    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(x_train)):
            error = train_design[index] @ weights - y_train[index]
            weights -= learning_rate * 2.0 * error * train_design[index]
        train_mse = float(np.mean((train_design @ weights - y_train) ** 2))
        val_mse = float(np.mean((val_design @ weights - y_val) ** 2))
        history.append((epoch, train_mse, val_mse))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_weights = weights.copy()
    return best_weights, history, best_epoch


class LinearRegressionSGD:
    """Small model interface shared by the command-line script and experiments."""

    def __init__(self, epochs=200, learning_rate=0.001, seed=42):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, x_train, y_train, x_val, y_val):
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        self.target_mean = float(y_train.mean())
        self.target_scale = float(y_train.std()) or 1.0
        self.weights, scaled_history, self.best_epoch = train_sgd(
            x_train, (y_train - self.target_mean) / self.target_scale,
            x_val, (y_val - self.target_mean) / self.target_scale,
            self.epochs, self.learning_rate, self.seed,
        )
        self.history = pd.DataFrame(
            [(epoch, train_mse * self.target_scale**2,
              val_mse * self.target_scale**2)
             for epoch, train_mse, val_mse in scaled_history],
            columns=["epoch", "train_mse", "validation_mse"],
        )
        return self

    def predict(self, features):
        return (self.weights[0] + features @ self.weights[1:]) * self.target_scale + self.target_mean

    def save(self, path, preprocessing):
        np.savez(
            path, weights=self.weights, columns=np.array(preprocessing.columns),
            drop_columns=np.array(preprocessing.drop_columns),
            medians=preprocessing.medians.to_numpy(),
            means=preprocessing.means.to_numpy(),
            scales=preprocessing.scales.to_numpy(),
            target_mean=self.target_mean, target_scale=self.target_scale,
            best_epoch=self.best_epoch,
        )


def main():
    hw1_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=hw1_dir / "train.csv")
    parser.add_argument("--kaggle-test-csv", type=Path, default=hw1_dir / "test.csv")
    parser.add_argument("--output-dir", type=Path, help="Save weights, loss history, and submission here")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-columns", nargs="+", default=NUMERIC_DROP_COLUMNS)
    args = parser.parse_args()

    frame = pd.read_csv(args.train_csv)
    if frame[TARGET].isna().any():
        raise ValueError("Training labels contain missing values")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(frame))
    train_end = round(0.70 * len(frame))
    val_end = train_end + round(0.15 * len(frame))
    train_rows = frame.iloc[indices[:train_end]]
    val_rows = frame.iloc[indices[train_end:val_end]]
    test_rows = frame.iloc[indices[val_end:]]

    preprocessing = NumericPreprocessor(args.drop_columns).fit(train_rows)
    x_train = preprocessing.transform(train_rows)
    x_val = preprocessing.transform(val_rows)
    x_test = preprocessing.transform(test_rows)
    y_train = train_rows[TARGET].to_numpy(dtype=np.float64)
    y_val = val_rows[TARGET].to_numpy(dtype=np.float64)
    y_test = test_rows[TARGET].to_numpy(dtype=np.float64)

    model = LinearRegressionSGD(args.epochs, args.learning_rate, args.seed)
    model.fit(x_train, y_train, x_val, y_val)

    for name, actual, features in (
        ("Train", y_train, x_train),
        ("Validation", y_val, x_val),
        ("Held-out test", y_test, x_test),
    ):
        mse, r_squared = scores(actual, model.predict(features))
        print(f"{name}: n={len(actual)}, MSE={mse:.3f}, R²={r_squared:.4f}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(args.output_dir / "linear_regression_weights.npz", preprocessing)
        model.history.to_csv(args.output_dir / "linear_regression_loss.csv", index=False)
        kaggle_test = pd.read_csv(args.kaggle_test_csv)
        pd.DataFrame({
            "id": kaggle_test["id"],
            TARGET: model.predict(preprocessing.transform(kaggle_test)),
        }).to_csv(args.output_dir / "submission.csv", index=False)
        print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
