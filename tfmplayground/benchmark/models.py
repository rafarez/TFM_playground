from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np

NEURALK_API_KEY="nk_live_CjUu-MWAuLE_De93aym7vcIk-tqBxY53ZvKOi9gyhiQ"

class BaseEvaluatedModel(ABC):
    """Interface the benchmark harness expects from any model it evaluates."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class NanoTabPFNEvalModel(BaseEvaluatedModel):
    """Wraps `NanoTabPFNClassifier`. Works for both the released checkpoint
    and any checkpoint produced by `pretrain_classification.py`."""

    name = "nanotabpfn"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        num_mem_chunks: int = 8,
    ):
        from tfmplayground import NanoTabPFNClassifier

        self._model = NanoTabPFNClassifier(
            model=checkpoint,
            device=device,
            num_mem_chunks=num_mem_chunks,
        )
        self._model.model.eval()

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class SeldonEvalModel(BaseEvaluatedModel):
    """Wraps the Neuralk Seldon API model. `predict_proba` is probed at runtime
    because the public snippet only documents `predict` — if it's missing we
    fall back to one-hot so AUC/log-loss remain computable."""

    name = "seldon"

    def __init__(self, api_key: str = NEURALK_API_KEY):
        from neuralk import Seldon

        self._model = Seldon(api_key=api_key)
        self._classes_: np.ndarray | None = None

    def fit(self, X, y):
        y = np.asarray(y)
        self._classes_ = np.unique(y)
        self._model.fit(X, y)

    def predict_proba(self, X):
        if hasattr(self._model, "predict_proba"):
            return np.asarray(self._model.predict_proba(X))
        warnings.warn(
            "Seldon model does not expose predict_proba; "
            "falling back to one-hot of predict(). AUC and log-loss will be degraded.",
            RuntimeWarning,
            stacklevel=2,
        )
        preds = np.asarray(self._model.predict(X))
        classes = self._classes_ if self._classes_ is not None else np.unique(preds)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        proba = np.zeros((len(preds), len(classes)))
        for i, p in enumerate(preds):
            if p in class_to_idx:
                proba[i, class_to_idx[p]] = 1.0
        return proba


def build_model(args) -> BaseEvaluatedModel:
    """Instantiates a `BaseEvaluatedModel` from argparse CLI arguments."""
    if args.model == "nanotabpfn":
        return NanoTabPFNEvalModel(
            checkpoint=args.checkpoint,
            device=args.device,
            num_mem_chunks=args.num_mem_chunks,
        )
    if args.model == "seldon":
        key = args.api_key or os.environ.get("NEURALK_API_KEY")
        if not key:
            raise ValueError(
                "Seldon requires an API key: pass --api_key or set the NEURALK_API_KEY env var."
            )
        return SeldonEvalModel(api_key=key)
    raise ValueError(f"Unknown model: {args.model!r}")
