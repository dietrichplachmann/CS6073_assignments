"""Generate Kaggle predictions from an existing experiment checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import MODEL_REGISTRY


HW2_DIR = Path(__file__).resolve().parent
# Highest validation accuracy among the three completed learning-rate suites.
RUN_DIR = HW2_DIR / "results" / "lr0.001"
TEST_CSV = HW2_DIR / "test.csv"
TEMPLATE_CSV = HW2_DIR / "sample_submission.csv"
OUTPUT_CSV = HW2_DIR / "submission.csv"
BATCH_SIZE = 128
DEVICE = "auto"


def generate_submission(*, run_dir, test_csv, template_csv, output_csv,
                        batch_size=128, device="auto", model_name=None,
                        overwrite=False):
    """Load saved weights and predict in original CSV order, without training.

    model_name defaults to the run's validation-selected best_model.json entry.
    The template defines the required row count and one-based ImageId values.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    run_dir, output_csv = Path(run_dir).resolve(), Path(output_csv).resolve()
    metadata_path = output_csv.with_suffix(".metadata.json")
    protected_paths = {Path(test_csv).resolve(), Path(template_csv).resolve()}
    if output_csv in protected_paths or metadata_path in protected_paths:
        raise ValueError("Submission output must differ from the test CSV and template.")
    if not overwrite and (output_csv.exists() or metadata_path.exists()):
        raise FileExistsError(f"Output already exists: {output_csv}. Use --overwrite to replace it.")
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device == "auto" else torch.device(device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment.")

    if model_name is None:
        selection = json.loads((run_dir / "best_model.json").read_text(encoding="utf-8"))
        model_name = selection["model"]
        weights_path = (run_dir / selection["weights"]).resolve()
    else:
        weights_path = run_dir / model_name / "best_weights.pt"
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}.")
    if not weights_path.is_relative_to(run_dir):
        raise ValueError("The saved checkpoint path must be inside the selected run directory.")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != model_name:
        raise ValueError("The saved checkpoint model does not match the requested architecture.")
    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    frame = pd.read_csv(test_csv)
    pixel_columns = [f"pixel{i}" for i in range(784)]
    if len(frame.columns) != 784 or set(frame.columns) != set(pixel_columns):
        raise ValueError("test.csv must contain exactly pixel0 through pixel783, without labels.")
    if frame.empty or frame.isna().any().any():
        raise ValueError("test.csv must contain images with no missing values.")
    pixels = frame[pixel_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Test pixels must be finite and between 0 and 255.")
    template = pd.read_csv(template_csv)
    if list(template.columns) != ["ImageId", "Label"] or len(template) != len(frame):
        raise ValueError("The template must have ImageId/Label columns and one row per test image.")
    ids = np.arange(1, len(frame) + 1, dtype=np.int64)
    if not np.array_equal(template["ImageId"].to_numpy(), ids):
        raise ValueError("Template ImageId values must run from 1 through the test row count.")
    # Match training exactly: floating-point [0, 1] pixels, one channel, 28 x 28.
    images = torch.from_numpy(pixels.reshape(-1, 1, 28, 28) / 255.0)
    predictions = []
    print(f"Using {model_name}, epoch {checkpoint['epoch']}, "
          f"learning rate {checkpoint['learning_rate']}, device {device}.", flush=True)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            logits = model(batch)
            if logits.shape != (len(batch), 10) or not torch.isfinite(logits).all():
                raise RuntimeError("The model returned invalid digit scores.")
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            if start // batch_size % 25 == 0 or start + batch_size >= len(images):
                print(f"Predicted {min(start + batch_size, len(images)):,}/{len(images):,} images.",
                      flush=True)
    labels = np.concatenate(predictions).astype(np.int64)
    if len(labels) != len(template) or not np.isin(labels, np.arange(10)).all():
        raise RuntimeError("Submission must have one digit from 0 through 9 per test image.")
    result = pd.DataFrame({"ImageId": ids, "Label": labels})
    metadata = {"run_dir": str(run_dir), "model": model_name,
                "checkpoint": str(weights_path), "epoch": checkpoint["epoch"],
                "learning_rate": checkpoint["learning_rate"],
                "validation_metrics": checkpoint["validation_metrics"],
                "checkpoint_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "test_csv": str(Path(test_csv).resolve()), "rows": len(result),
                "label_counts": {str(k): int(v) for k, v in result.Label.value_counts().sort_index().items()},
                "output_csv": str(output_csv)}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(result):,} predictions to {output_csv}", flush=True)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--template-csv", type=Path, default=TEMPLATE_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_submission(run_dir=args.run_dir, test_csv=args.test_csv,
                        template_csv=args.template_csv, output_csv=args.output,
                        batch_size=args.batch_size, device=args.device,
                        model_name=args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
