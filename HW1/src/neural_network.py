"""Fully connected neural network regression with NumPy and mini-batch SGD."""

import numpy as np
import pandas as pd


class NeuralNetworkRegressor:
    """ReLU hidden layers, one linear output, and MSE or Huber loss."""

    def __init__(self, hidden_layers, epochs=250, learning_rate=0.005,
                 batch_size=32, seed=42, loss="mse", huber_delta=1.0):
        self.hidden_layers = tuple(hidden_layers)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.loss = loss
        self.huber_delta = huber_delta

    def _forward(self, features):
        activations = [features]
        pre_activations = []
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            values = activations[-1] @ weights + biases
            pre_activations.append(values)
            activations.append(np.maximum(values, 0.0))
        output = activations[-1] @ self.weights[-1] + self.biases[-1]
        activations.append(output)
        return activations, pre_activations

    def fit(self, x_train, y_train, x_val, y_val):
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive layer widths")
        if self.epochs < 1 or self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("epochs, learning rate, and batch size must be positive")
        if self.loss not in {"mse", "huber"} or self.huber_delta <= 0:
            raise ValueError("loss must be 'mse' or 'huber', with positive huber_delta")
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ValueError("Features must be finite after preprocessing")

        self.target_mean = float(y_train.mean())
        self.target_scale = float(y_train.std()) or 1.0
        scaled_y = ((y_train - self.target_mean) / self.target_scale).reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        dimensions = (x_train.shape[1], *self.hidden_layers, 1)
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])
        ]
        self.biases = [np.zeros(fan_out) for fan_out in dimensions[1:]]
        history = []
        best_val_mse = float("inf")
        self.best_epoch = 0
        best_weights = None
        best_biases = None

        for epoch in range(1, self.epochs + 1):
            shuffled = rng.permutation(len(x_train))
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                activations, pre_activations = self._forward(x_train[batch])
                residual = activations[-1] - scaled_y[batch]
                if self.loss == "huber":
                    # Twice the usual Huber loss matches MSE's local curvature.
                    residual = np.clip(residual, -self.huber_delta, self.huber_delta)
                delta = 2.0 * residual / len(batch)
                gradients_w = [None] * len(self.weights)
                gradients_b = [None] * len(self.biases)
                for layer in range(len(self.weights) - 1, -1, -1):
                    gradients_w[layer] = activations[layer].T @ delta
                    gradients_b[layer] = delta.sum(axis=0)
                    if layer:
                        delta = (delta @ self.weights[layer].T) * (
                            pre_activations[layer - 1] > 0
                        )
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.learning_rate * gradients_w[layer]
                    self.biases[layer] -= self.learning_rate * gradients_b[layer]

            train_mse = float(np.mean((self.predict(x_train) - y_train) ** 2))
            val_mse = float(np.mean((self.predict(x_val) - y_val) ** 2))
            if not np.isfinite(train_mse + val_mse):
                raise FloatingPointError("Training diverged; lower the learning rate")
            history.append((epoch, train_mse, val_mse))
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                self.best_epoch = epoch
                best_weights = [weights.copy() for weights in self.weights]
                best_biases = [biases.copy() for biases in self.biases]

        self.weights = best_weights
        self.biases = best_biases
        self.history = pd.DataFrame(
            history, columns=["epoch", "train_mse", "validation_mse"],
        )
        return self

    def predict(self, features):
        output = self._forward(features)[0][-1].ravel()
        return output * self.target_scale + self.target_mean

    def save(self, path, preprocessing, input_columns=None):
        arrays = {
            "hidden_layers": np.array(self.hidden_layers),
            "columns": np.array(preprocessing.columns),
            "model_input_columns": np.array(
                input_columns if input_columns is not None else preprocessing.columns,
            ),
            "drop_columns": np.array(preprocessing.drop_columns),
            "medians": preprocessing.medians.to_numpy(),
            "means": preprocessing.means.to_numpy(),
            "scales": preprocessing.scales.to_numpy(),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "best_epoch": self.best_epoch,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
        }
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weights_{index}"] = weights
            arrays[f"biases_{index}"] = biases
        np.savez(path, **arrays)
