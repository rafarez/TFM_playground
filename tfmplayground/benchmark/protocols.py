from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def iter_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    protocol: str,
    seed: int,
    test_size: float = 0.2,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields `(train_idx, test_idx)` pairs for the requested protocol.

    - `fixed`: a single stratified 80/20 split (size controlled by `test_size`).
    - `5fold`: stratified 5-fold cross-validation.

    Falls back to non-stratified splitting when a class has fewer than 2
    samples, which stratification requires.
    """
    idx = np.arange(len(y))
    can_stratify = np.min(np.bincount(y)) >= 2 if y.dtype.kind in "iu" else True

    if protocol == "fixed":
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=y if can_stratify else None,
        )
        yield train_idx, test_idx
        return

    if protocol == "5fold":
        if can_stratify:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in kf.split(np.zeros(len(y)), y):
                yield train_idx, test_idx
        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in kf.split(idx):
                yield train_idx, test_idx
        return

    raise ValueError(f"Unknown protocol: {protocol!r}")
