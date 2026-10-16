"""Edit the experiment settings below, then run: python HW2/main.py."""

from pathlib import Path

from experiments import run_suite


HW2_DIR = Path(__file__).resolve().parent
TRAIN_CSV = HW2_DIR / "train.csv"
OUTPUT_DIR = HW2_DIR / "results"

# Run again with 0.1 and 0.01 to fill the assignment's learning-rate tables.
LEARNING_RATE = 0.1
EPOCHS = 20
BATCH_SIZE = 128
ENABLED_MODELS = ("DNN", "ConvNet", "VGG", "ResNet")
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10  # The remainder is the held-out local test set.
SEED = 42
DEVICE = "auto"  # "auto", "cpu", or "cuda"; auto uses CUDA when available.
NUM_WORKERS = 0  # Safe default on Windows.
PATIENCE = 5  # Stop after this many epochs without better validation accuracy.
MAX_SAMPLES = None  # Set to e.g. 1000 for a quick smoke run; None uses all rows.


if __name__ == "__main__":
    run_suite(
        train_csv=TRAIN_CSV,
        output_root=OUTPUT_DIR,
        model_names=ENABLED_MODELS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
        device=DEVICE,
        num_workers=NUM_WORKERS,
        patience=PATIENCE,
        max_samples=MAX_SAMPLES,
    )
