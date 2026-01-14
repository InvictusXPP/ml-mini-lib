#Dense, Dropout, BatchNote

from typing import Callable
from .backend import get_xp


class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        backend: str = "cpu",
        activation: Callable = None,
        name: str = None,
    ):
        self.backend = backend
        self.xp = get_xp(backend)

        self.W = self.xp.zeros(
            (in_features, out_features), dtype=self.xp.float32
        )
        self.b = self.xp.zeros(
            (1, out_features), dtype=self.xp.float32
        )

        self.activation = activation
        self.name = name or f"Dense_{in_features}x{out_features}"

    def init_weights(self, seed: int = 42, scale: float = 0.5):
        if self.xp.__name__ == "numpy":
            import numpy as np
            rng = np.random.RandomState(seed)
            self.W = (rng.randn(*self.W.shape) * scale).astype(np.float32)
            self.b = np.zeros_like(self.b)
        else:
            try:
                self.xp.random.seed(seed)
            except Exception:
                pass
            self.W = self.xp.random.randn(*self.W.shape).astype(
                self.xp.float32
            ) * scale
            self.b = self.xp.zeros_like(self.b)

    def forward(self, X):
        Z = X @ self.W + self.b
        if self.activation is None:
            return Z
        return self.activation(Z, self.xp)

    def __repr__(self):
        return f"<Dense {self.name} ({self.W.shape[0]} -> {self.W.shape[1]})>"


class Dropout:
    def __init__(self, p: float = 0.5, backend: str = "cpu"):
        self.p = p
        self.backend = backend
        self.xp = get_xp(backend)
        self.mask = None

    def forward(self, X, train: bool = True):
        if not train or self.p <= 0.0:
            return X

        self.mask = (
            (self.xp.random.rand(*X.shape) > self.p)
            .astype(X.dtype)
            / (1.0 - self.p)
        )
        return X * self.mask

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask


class BatchNormSimple:
    def __init__(
        self,
        dim: int,
        backend: str = "cpu",
        eps: float = 1e-5,
    ):
        self.backend = backend
        self.xp = get_xp(backend)

        self.gamma = self.xp.ones((1, dim), dtype=self.xp.float32)
        self.beta = self.xp.zeros((1, dim), dtype=self.xp.float32)
        self.eps = eps

        # cache
        self.X_centered = None
        self.std = None

    def forward(self, X):
        mu = self.xp.mean(X, axis=0, keepdims=True)
        var = self.xp.var(X, axis=0, keepdims=True)

        self.X_centered = X - mu
        self.std = self.xp.sqrt(var + self.eps)

        X_norm = self.X_centered / self.std
        return self.gamma * X_norm + self.beta

    def backward(self, dY):
        """
        Zwraca:
        - dx
        - dgamma
        - dbeta
        """
        N = dY.shape[0]

        dgamma = self.xp.sum(
            dY * (self.X_centered / self.std),
            axis=0,
            keepdims=True,
        )
        dbeta = self.xp.sum(dY, axis=0, keepdims=True)

        dx = (
            (1.0 / N)
            * (1.0 / self.std)
            * (
                N * dY
                - self.xp.sum(dY, axis=0, keepdims=True)
                - (self.X_centered / (self.std ** 2))
                * self.xp.sum(
                    dY * self.X_centered,
                    axis=0,
                    keepdims=True,
                )
            )
        )

        return dx, dgamma, dbeta
