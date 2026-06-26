"""Dependency-free ML primitives (pure NumPy)."""
from __future__ import annotations

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = max(1, int(len(X) * test_size))
    return X[idx[n:]], X[idx[:n]], y[idx[n:]], y[idx[:n]]


class Standardizer:
    def fit(self, X: np.ndarray) -> Standardizer:
        X = np.asarray(X, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mu_) / self.sd_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class LogisticRegression:
    def __init__(
        self,
        lr: float = 0.2,
        epochs: int = 400,
        l2: float = 1e-3,
        seed: int = 0,
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        pos = max(y.sum(), 1.0)
        neg = max((1 - y).sum(), 1.0)
        sample_weight = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w_ + self.b_)
            err = (p - y) * sample_weight
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)

    def predict(self, X: np.ndarray, t: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= t).astype(int)


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        A = Xb.T @ Xb + self.alpha * np.eye(Xb.shape[1])
        A[0, 0] -= self.alpha
        self.coef_ = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)]) @ self.coef_


def roc_auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(y) == np.asarray(pred)).mean())


def f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2)))


def mape(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.abs(y) > 1e-8
    return float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100)
