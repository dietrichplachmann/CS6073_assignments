"""Run HW1 models on the same data split and compare their results.

Edit the settings below, then run this file. New models can be added to
MODEL_REGISTRY if they implement fit, predict, save, and history like
LinearRegressionSGD.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from linear_regression import LinearRegressionSGD, scores
from neural_network import NeuralNetworkRegressor
from plot_losses import plot_loss_curves
from preprocess import NumericPreprocessor, TARGET
from topological_features import FEATURE_NAMES, LocalPersistenceFeatures


# Experiment settings: change these and run main.py again.
HW1_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = HW1_DIR / "train.csv"
KAGGLE_TEST_CSV = HW1_DIR / "test.csv"
OUTPUT_DIR = HW1_DIR / "results"
SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
DROP_COLUMNS = ("id", "Geography", "binnedInc", "PctSomeCol18_24")
ENABLED_MODELS = (
    "linear_regression",
    "dnn_8",
    "dnn_16_8",
    "dnn_16_8_4",
    "dnn_16_8_4_huber",
    "dnn_30_16_8_4",
    "tda_dnn_16_8_4",
)
CREATE_KAGGLE_SUBMISSION = True

LINEAR_EPOCHS = 200
LINEAR_LEARNING_RATE = 0.001
DNN_EPOCHS = 250
DNN_LEARNING_RATE = 0.005
DNN_BATCH_SIZE = 32
HUBER_DELTA = 1.0  # Measured in standardized target units.
TDA_NEIGHBORS = 12

# Add future models here, then put their names in ENABLED_MODELS.
MODEL_REGISTRY = {
    "linear_regression": lambda: LinearRegressionSGD(
        epochs=LINEAR_EPOCHS, learning_rate=LINEAR_LEARNING_RATE, seed=SEED,
    ),
    "dnn_8": lambda: NeuralNetworkRegressor(
        (8,), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
    ),
    "dnn_16_8": lambda: NeuralNetworkRegressor(
        (16, 8), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
    ),
    "dnn_16_8_4": lambda: NeuralNetworkRegressor(
        (16, 8, 4), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
    ),
    "dnn_16_8_4_huber": lambda: NeuralNetworkRegressor(
        (16, 8, 4), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
        loss="huber", huber_delta=HUBER_DELTA,
    ),
    "dnn_30_16_8_4": lambda: NeuralNetworkRegressor(
        (30, 16, 8, 4), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
    ),
    "tda_dnn_16_8_4": lambda: NeuralNetworkRegressor(
        (16, 8, 4), DNN_EPOCHS, DNN_LEARNING_RATE, DNN_BATCH_SIZE, SEED,
    ),
}

# Each model uses the standard inputs unless listed here.
MODEL_FEATURE_SETS = {"tda_dnn_16_8_4": "tda"}


def split_labeled_data(frame):
    if not 0 < TRAIN_FRACTION < 1 or not 0 < VALIDATION_FRACTION < 1:
        raise ValueError("Training and validation fractions must be between 0 and 1")
    if TRAIN_FRACTION + VALIDATION_FRACTION >= 1:
        raise ValueError("Leave a positive fraction for the held-out test set")
    indices = np.random.default_rng(SEED).permutation(len(frame))
    train_end = round(TRAIN_FRACTION * len(frame))
    val_end = train_end + round(VALIDATION_FRACTION * len(frame))
    return {
        "train": frame.iloc[indices[:train_end]],
        "validation": frame.iloc[indices[train_end:val_end]],
        "test": frame.iloc[indices[val_end:]],
    }


def run_experiments():
    if not ENABLED_MODELS:
        raise ValueError("Add at least one model name to ENABLED_MODELS")
    unknown = set(ENABLED_MODELS) - set(MODEL_REGISTRY)
    if unknown:
        raise ValueError(f"Models missing from MODEL_REGISTRY: {sorted(unknown)}")
    if len(set(ENABLED_MODELS)) != len(ENABLED_MODELS):
        raise ValueError("ENABLED_MODELS contains duplicate names")

    frame = pd.read_csv(TRAIN_CSV)
    if frame[TARGET].isna().any():
        raise ValueError("Training labels contain missing values")
    rows = split_labeled_data(frame)
    preprocessing = NumericPreprocessor(DROP_COLUMNS).fit(rows["train"])
    features = {name: preprocessing.transform(part) for name, part in rows.items()}
    feature_sets = {"standard": features}
    topology = None
    if any(MODEL_FEATURE_SETS.get(name) == "tda" for name in ENABLED_MODELS):
        topology = LocalPersistenceFeatures(TDA_NEIGHBORS)
        tda_features = {
            "train": np.column_stack((
                features["train"], topology.fit_transform(features["train"]),
            )),
            "validation": np.column_stack((
                features["validation"], topology.transform(features["validation"]),
            )),
            "test": np.column_stack((
                features["test"], topology.transform(features["test"]),
            )),
        }
        feature_sets["tda"] = tda_features
    labels = {
        name: part[TARGET].to_numpy(dtype=np.float64)
        for name, part in rows.items()
    }
    print("Split:", ", ".join(f"{name}={len(part)}" for name, part in rows.items()))
    print(f"Numeric features: {len(preprocessing.columns)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if topology is not None:
        topology.save(OUTPUT_DIR / "tda_transform.npz")
    results = []
    histories = []
    best_name = None
    best_model = None
    best_feature_set = None
    best_validation_mse = float("inf")

    for name in ENABLED_MODELS:
        feature_set_name = MODEL_FEATURE_SETS.get(name, "standard")
        model_features = feature_sets[feature_set_name]
        model = MODEL_REGISTRY[name]()
        model.fit(
            model_features["train"], labels["train"],
            model_features["validation"], labels["validation"],
        )
        row = {"model": name, "feature_set": feature_set_name,
               "training_loss": getattr(model, "loss", "mse"),
               "best_epoch": model.best_epoch}
        for split in ("train", "validation", "test"):
            predictions = model.predict(model_features[split])
            mse, r_squared = scores(
                labels[split], predictions,
            )
            row[f"{split}_mse"] = mse
            row[f"{split}_mae"] = float(np.mean(np.abs(labels[split] - predictions)))
            row[f"{split}_r2"] = r_squared
        results.append(row)
        history = model.history.copy()
        history.insert(0, "model", name)
        histories.append(history)
        if feature_set_name == "tda":
            model.save(
                OUTPUT_DIR / f"{name}_weights.npz", preprocessing,
                input_columns=[*preprocessing.columns, *FEATURE_NAMES],
            )
        else:
            model.save(OUTPUT_DIR / f"{name}_weights.npz", preprocessing)
        print(
            f"{name}: best epoch={model.best_epoch}, "
            f"validation MSE={row['validation_mse']:.3f}, "
            f"validation R²={row['validation_r2']:.4f}, "
            f"test MSE={row['test_mse']:.3f}, test R²={row['test_r2']:.4f}"
        )
        if row["validation_mse"] < best_validation_mse:
            best_name, best_model = name, model
            best_feature_set = feature_set_name
            best_validation_mse = row["validation_mse"]

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    loss_history = pd.concat(histories, ignore_index=True)
    loss_history.to_csv(OUTPUT_DIR / "loss_history.csv", index=False)
    plot_path = plot_loss_curves(loss_history, OUTPUT_DIR / "loss_curves.png")
    print(f"Loss plot: {plot_path}")

    if CREATE_KAGGLE_SUBMISSION:
        kaggle_test = pd.read_csv(KAGGLE_TEST_CSV)
        kaggle_features = preprocessing.transform(kaggle_test)
        if best_feature_set == "tda":
            kaggle_features = np.column_stack((
                kaggle_features, topology.transform(kaggle_features),
            ))
        pd.DataFrame({
            "id": kaggle_test["id"],
            TARGET: best_model.predict(kaggle_features),
        }).to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Submission model: {best_name}")
    print(f"Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiments()
