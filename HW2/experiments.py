"""Run comparable experiments and save metrics, checkpoints, and report plots."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve

from data import load_splits
from models import MODEL_REGISTRY
from training import evaluate, make_loaders, seed_everything, train_model


def plot_history(history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            ax.plot(history["epoch"], history[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_roc(labels, probabilities, destination):
    fig, ax = plt.subplots(figsize=(7, 6))
    for digit in range(10):
        fpr, tpr, _ = roc_curve(labels == digit, probabilities[:, digit])
        ax.plot(fpr, tpr, label=f"{digit}: AUC={auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Held-out test ROC curves (one vs rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_suite(*, train_csv, output_root, model_names, learning_rate=0.001,
              epochs=20, batch_size=128, train_fraction=0.8,
              validation_fraction=0.1, seed=42, device="auto", num_workers=0,
              patience=5, max_samples=None):
    model_names = tuple(model_names)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Choose at least one model, without repeated names.")
    if set(model_names) - MODEL_REGISTRY.keys():
        raise ValueError(f"Available models: {', '.join(MODEL_REGISTRY)}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if min(epochs, batch_size, patience) < 1 or num_workers < 0:
        raise ValueError("epochs, batch_size, patience must be positive; workers cannot be negative.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment.")
    datasets, indices = load_splits(
        train_csv, train_fraction, validation_fraction, seed, max_samples,
    )
    run_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez(run_dir / "split_indices.npz", **indices)
    config = {"train_csv": str(Path(train_csv).resolve()), "model_names": model_names,
              "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size,
              "train_fraction": train_fraction, "validation_fraction": validation_fraction,
              "seed": seed, "device": str(device), "num_workers": num_workers,
              "patience": patience, "max_samples": max_samples,
              "split_sizes": {name: len(ds) for name, ds in datasets.items()},
              "checkpoint_selection": "validation accuracy; ties use lower validation loss",
              "f1_average": "macro", "auc_average": "macro", "auc_multiclass": "ovr",
              "torch_version": str(torch.__version__)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device}; splits: {config['split_sizes']}; output: {run_dir}", flush=True)
    results = []
    selected_name, selected_key, selected_labels, selected_probabilities = None, None, None, None
    for name in model_names:
        # Reset initialization and loader randomness for reproducible comparisons.
        seed_everything(seed)
        loaders = make_loaders(datasets, batch_size, num_workers, seed, device)
        model = MODEL_REGISTRY[name]().to(device)
        model_dir = run_dir / name
        model_dir.mkdir()
        history, best_epoch, validation = train_model(
            model, loaders, device, learning_rate, epochs, patience, model_dir, name,
        )
        test, labels, probabilities = evaluate(model, loaders["test"], device)
        row = {"model": name, "learning_rate": learning_rate, "best_epoch": best_epoch,
               "epochs_run": len(history),
               "parameters": sum(p.numel() for p in model.parameters()),
               **{f"selected_validation_{k}": v for k, v in validation.items()},
               **{f"test_{k}": v for k, v in test.items()}}
        for metric in ("f1_macro", "auc_macro_ovr"):
            column = f"validation_{metric}"
            index = history[column].idxmax()
            row[f"best_{column}"] = float(history.loc[index, column])
            row[f"best_{column}_epoch"] = int(history.loc[index, "epoch"])
        results.append(row)
        pd.DataFrame(results).to_csv(run_dir / "summary.csv", index=False)
        np.savez_compressed(model_dir / "test_predictions.npz", labels=labels,
                            probabilities=probabilities, row_indices=indices["test"])
        plot_history(history, model_dir / "learning_curves.png")
        key = (validation["accuracy"], -validation["loss"])
        if selected_key is None or key > selected_key:
            selected_name, selected_key = name, key
            selected_labels, selected_probabilities = labels, probabilities
        print(f"{name}: TEST accuracy={test['accuracy']:.4f}, "
              f"F1={test['f1_macro']:.4f}, AUC={test['auc_macro_ovr']:.4f}; "
              f"best validation F1={row['best_validation_f1_macro']:.4f}, "
              f"AUC={row['best_validation_auc_macro_ovr']:.4f}", flush=True)
        del model, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()
    plot_roc(selected_labels, selected_probabilities, run_dir / "best_model_test_roc.png")
    (run_dir / "best_model.json").write_text(json.dumps({
        "model": selected_name, "selection": "validation accuracy, then validation loss",
        "weights": f"{selected_name}/best_weights.pt",
    }, indent=2), encoding="utf-8")
    summary = pd.DataFrame(results)
    print("\n" + summary[["model", "test_accuracy", "test_f1_macro", "test_auc_macro_ovr"]].to_string(index=False))
    print(f"Selected model: {selected_name}. Results saved in {run_dir}", flush=True)
    return run_dir, summary
