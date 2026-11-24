"""Deterministic synthetic datasets for the lifecycle demo.

Generates a tabular classification problem (predicting subscription churn)
from a seeded RNG, plus two drifted variants used by the monitoring stage:

* ``covariate drift``: one feature's distribution is shifted in live data;
* ``performance decay``: label noise reduces separability.

Everything is seeded, so CI, tests, and the README numbers are reproducible.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from mlops_lifecycle.models import DatasetHandle

FloatArray = NDArray[np.float64]


def make_dataset(
    n_samples: int = 4000,
    n_features: int = 8,
    seed: int = 42,
    label_noise: float = 0.0,
    drift_feature: int | None = None,
    drift_scale: float = 1.0,
) -> tuple[FloatArray, FloatArray, DatasetHandle]:
    """Generate one (X, y) dataset with an attached fingerprint.

    Args:
        label_noise: fraction of flipped labels (0 = clean separable data).
        drift_feature: index of a feature to shift (simulates covariate drift).
        drift_scale: how hard that feature is shifted (standard deviations).
    """
    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # Linear score + interaction so a linear model is good but not perfect.
    weights = rng.normal(loc=0.0, scale=1.0, size=n_features)
    logits = X @ weights + 0.8 * X[:, 0] * X[:, 1]
    y = (logits > np.median(logits)).astype(np.float64)

    if label_noise > 0.0:
        flip = rng.random(n_samples) < label_noise
        y = np.where(flip, 1.0 - y, y)

    if drift_feature is not None:
        if not 0 <= drift_feature < n_features:
            msg = f"drift_feature index out of range: {drift_feature}"
            raise ValueError(msg)
        X[:, drift_feature] += drift_scale * 2.5

    handle = DatasetHandle(
        name=f"synthetic-churn-n{n_samples}-f{n_features}-s{seed}"
        f"{'-drift' if drift_feature is not None else ''}"
        f"{'-noise' if label_noise else ''}",
        n_samples=n_samples,
        n_features=n_features,
        sha8=_fingerprint(X, y),
    )
    return X, y, handle


def _fingerprint(X: FloatArray, y: FloatArray) -> str:
    """Stable 8-hex fingerprint of the data content (not object identity)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:8]


def train_val_test_split(
    X: FloatArray,
    y: FloatArray,
    handle: DatasetHandle,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[
    tuple[FloatArray, FloatArray],
    tuple[FloatArray, FloatArray],
    tuple[FloatArray, FloatArray],
]:
    """Split deterministically into train/val/test tuples of (X, y)."""
    if not 0.0 < val_fraction < 0.5 or not 0.0 < test_fraction < 0.5:
        msg = "fractions must be in (0, 0.5)"
        raise ValueError(msg)
    if val_fraction + test_fraction >= 0.9:
        msg = "val + test must stay below 0.9"
        raise ValueError(msg)

    rng = np.random.default_rng(seed + 1)
    indices = rng.permutation(len(X))
    n_val = int(len(indices) * val_fraction)
    n_test = int(len(indices) * test_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    def take(idx: NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        return X[idx], y[idx]

    _ = handle  # fingerprints travel separately; split is pure
    return take(train_idx), take(val_idx), take(test_idx)


def simulate_live_batch(
    *,
    seed: int = 777,
    n_samples: int = 1200,
    n_features: int = 8,
    drift_feature: int | None = None,
    drift_scale: float = 1.0,
    label_noise: float = 0.0,
) -> tuple[FloatArray, FloatArray, DatasetHandle]:
    """A 'live' batch, optionally drifted, for the monitoring stage."""
    return make_dataset(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
        label_noise=label_noise,
        drift_feature=drift_feature,
        drift_scale=drift_scale,
    )
