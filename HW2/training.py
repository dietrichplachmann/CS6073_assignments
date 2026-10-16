"""Shared training, probability-based metrics, and validation checkpoint selection."""

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loaders(datasets, batch_size, num_workers, seed, device):
    return {
        name: DataLoader(
            dataset, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            generator=torch.Generator().manual_seed(seed),
        )
        for name, dataset in datasets.items()
    }


def classification_scores(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=np.arange(10),
                                    average="macro", zero_division=0)),
        "auc_macro_ovr": float(roc_auc_score(
            labels, probabilities, labels=np.arange(10), multi_class="ovr", average="macro",
        )),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    all_labels, all_probabilities = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        all_labels.append(labels.cpu().numpy())
        # AUC must use class probabilities, rather than hard digit predictions.
        all_probabilities.append(logits.softmax(dim=1).cpu().numpy())
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite predictions; try a smaller learning rate.")
    metrics = classification_scores(labels, probabilities)
    metrics["loss"] = loss_sum / len(labels)
    return metrics, labels, probabilities


def train_model(model, loaders, device, learning_rate, epochs, patience, output_dir,
                model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    best_epoch, stale_epochs = 0, 0
    checkpoint_path = output_dir / "best_weights.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, samples = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name}: non-finite loss; try a smaller learning rate.")
            loss.backward()
            optimizer.step()
            samples += len(labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        validation, _, _ = evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": total_loss / samples,
               "train_accuracy": correct / samples,
               **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss = validation["accuracy"], validation["loss"]
            best_epoch, stale_epochs = epoch, 0
            torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                        "epoch": epoch, "learning_rate": learning_rate,
                        "validation_metrics": validation}, checkpoint_path)
        else:
            stale_epochs += 1
        print(f"{model_name} epoch {epoch}/{epochs}: "
              f"train acc={row['train_accuracy']:.4f}, val acc={validation['accuracy']:.4f}, "
              f"val F1={validation['f1_macro']:.4f}, val AUC={validation['auc_macro_ovr']:.4f}",
              flush=True)
        if stale_epochs >= patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return pd.DataFrame(history), best_epoch, checkpoint["validation_metrics"]
