"""Validate MNIST CSV data and produce shared, stratified data splits."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_splits(path, train_fraction=0.8, validation_fraction=0.1, seed=42,
                max_samples=None):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Training, validation, and test fractions must be positive and sum to 1.")
    frame = pd.read_csv(path)
    pixels = [f"pixel{i}" for i in range(784)]
    if set(frame.columns) != {"label", *pixels}:
        raise ValueError("Expected label and pixel0 through pixel783 exactly.")
    if frame.isna().any().any():
        raise ValueError("The labeled CSV contains missing values.")
    x = frame[pixels].to_numpy(dtype=np.float32)
    raw_y = frame["label"].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError("Pixel values must be finite and between 0 and 255.")
    if not np.isfinite(raw_y).all() or not np.isin(raw_y, np.arange(10)).all():
        raise ValueError("Labels must be integers from 0 through 9.")
    y = raw_y.astype(np.int64)
    if set(y) != set(range(10)):
        raise ValueError("All ten digit classes must be present for multiclass AUC.")
    indices = np.arange(len(y))
    if max_samples is not None:
        if not isinstance(max_samples, int) or not 0 < max_samples <= len(y):
            raise ValueError("max_samples must be a positive integer up to the CSV row count.")
        if max_samples < len(y):
            indices, _ = train_test_split(
                indices, train_size=max_samples, stratify=y, random_state=seed,
            )
    train_idx, remainder = train_test_split(
        indices, train_size=train_fraction, stratify=y[indices], random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        remainder, train_size=validation_fraction / (1 - train_fraction),
        stratify=y[remainder], random_state=seed,
    )
    split_indices = {"train": train_idx, "validation": val_idx, "test": test_idx}
    datasets = {}
    for name, idx in split_indices.items():
        if set(y[idx]) != set(range(10)):
            raise ValueError(f"{name} lacks a digit class; increase max_samples or split size.")
        images = torch.from_numpy(x[idx].reshape(-1, 1, 28, 28) / 255.0)
        datasets[name] = TensorDataset(images, torch.from_numpy(y[idx]))
    return datasets, split_indices
