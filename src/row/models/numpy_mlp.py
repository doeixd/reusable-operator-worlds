"""Small dependency-light NumPy residual MLP for scratch controls."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ScratchResidualMLP:
    """A tanh residual MLP with AdamW, independently initialized per task."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.params: dict[str, Array] = {
            "W1": rng.normal(scale=np.sqrt(1.0 / input_dim), size=(input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(scale=1e-2, size=(hidden_dim, input_dim)),
            "b2": np.zeros(input_dim),
        }
        self.first = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = 0

    def predict(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        hidden = np.tanh(x @ self.params["W1"] + self.params["b1"])
        return np.tanh(x + hidden @ self.params["W2"] + self.params["b2"])

    def update(self, x: Array, y: Array) -> float:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        hidden = np.tanh(x @ self.params["W1"] + self.params["b1"])
        prediction = np.tanh(x + hidden @ self.params["W2"] + self.params["b2"])
        error = prediction - y
        loss = float(np.mean(np.square(error)))

        d_output = (2.0 / error.size) * error * (1.0 - np.square(prediction))
        gradients: dict[str, Array] = {
            "W2": hidden.T @ d_output,
            "b2": np.sum(d_output, axis=0),
        }
        d_hidden = (d_output @ self.params["W2"].T) * (1.0 - np.square(hidden))
        gradients["W1"] = x.T @ d_hidden
        gradients["b1"] = np.sum(d_hidden, axis=0)
        self._adamw_step(gradients)
        return loss

    def _adamw_step(self, gradients: dict[str, Array]) -> None:
        self.steps += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, parameter in self.params.items():
            gradient = gradients[name]
            self.first[name] = beta1 * self.first[name] + (1.0 - beta1) * gradient
            self.second[name] = beta2 * self.second[name] + (1.0 - beta2) * np.square(gradient)
            first_hat = self.first[name] / (1.0 - beta1**self.steps)
            second_hat = self.second[name] / (1.0 - beta2**self.steps)
            parameter *= 1.0 - self.learning_rate * self.weight_decay
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

