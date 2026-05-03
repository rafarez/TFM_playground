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
        max_features: int | None = None,
        n_feature_subsets: int | None = None,
    ):
        from tfmplayground import NanoTabPFNClassifier

        self._model = NanoTabPFNClassifier(
            model=checkpoint,
            device=device,
            num_mem_chunks=num_mem_chunks,
            max_features=max_features,
            n_feature_subsets=n_feature_subsets,
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


class MyNanoTabPFNEvalModel(BaseEvaluatedModel):
    """Wraps NanoTabPFNClassifier backed by MyNanoTabPFNModel (P1 + P2 mods).

    When a checkpoint is given its stored flags take precedence over the
    constructor arguments (the checkpoint was trained with specific flags).
    When no checkpoint is given a fresh model is built from the constructor
    arguments using the released baseline hyperparameters (d=192, 6 layers).
    """

    name = "mynanotabpfn"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        num_mem_chunks: int = 8,
        # P1
        target_aware: bool = False,
        random_perturbations: bool = False,
        target_encoder_use_embedding: bool = False,
        n_perturbation_samples: int = 1,
        # P2
        prenorm: bool = False,
        cls_compression: bool = False,
        n_cls_tokens: int = 2,
        n_stage1_layers: int = 3,
        n_stage2_layers: int = 3,
        icl_target_embedding: bool = False,
        # P3
        feature_grouping: bool = False,
        gated_residuals: bool = False,
        multi_layer_decoder: bool = False,
        decoder_layer_indices: list[int] | None = None,
        # Mod 2.6
        max_features: int | None = None,
        n_feature_subsets: int | None = None,
    ):
        from tfmplayground.interface import NanoTabPFNClassifier, init_model_from_state_dict_file
        from tfmplayground.models.my_models import MyNanoTabPFNModel

        if checkpoint is not None:
            model_instance = init_model_from_state_dict_file(checkpoint)
        else:
            model_instance = MyNanoTabPFNModel(
                embedding_size=192,
                num_attention_heads=4,
                mlp_hidden_size=768,
                num_layers=6,
                num_outputs=10,
                target_aware=target_aware,
                random_perturbations=random_perturbations,
                target_encoder_use_embedding=target_encoder_use_embedding,
                prenorm=prenorm,
                cls_compression=cls_compression,
                n_cls_tokens=n_cls_tokens,
                n_stage1_layers=n_stage1_layers,
                n_stage2_layers=n_stage2_layers,
                icl_target_embedding=icl_target_embedding,
                feature_grouping=feature_grouping,
                gated_residuals=gated_residuals,
                multi_layer_decoder=multi_layer_decoder,
                decoder_layer_indices=decoder_layer_indices,
            )

        self._model = NanoTabPFNClassifier(
            model=model_instance,
            device=device,
            num_mem_chunks=num_mem_chunks,
            n_perturbation_samples=n_perturbation_samples,
            max_features=max_features,
            n_feature_subsets=n_feature_subsets,
        )
        self._model.model.eval()

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def build_model(args) -> BaseEvaluatedModel:
    """Instantiates a `BaseEvaluatedModel` from argparse CLI arguments."""
    if args.model == "nanotabpfn":
        return NanoTabPFNEvalModel(
            checkpoint=args.checkpoint,
            device=args.device,
            num_mem_chunks=args.num_mem_chunks,
            max_features=args.max_features,
            n_feature_subsets=args.n_feature_subsets,
        )
    if args.model == "mynanotabpfn":
        return MyNanoTabPFNEvalModel(
            checkpoint=args.checkpoint,
            device=args.device,
            num_mem_chunks=args.num_mem_chunks,
            target_aware=args.target_aware,
            random_perturbations=args.random_perturbations,
            target_encoder_use_embedding=args.target_encoder_use_embedding,
            n_perturbation_samples=args.n_perturbation_samples,
            prenorm=args.prenorm,
            cls_compression=args.cls_compression,
            n_cls_tokens=args.n_cls_tokens,
            n_stage1_layers=args.n_stage1_layers,
            n_stage2_layers=args.n_stage2_layers,
            icl_target_embedding=args.icl_target_embedding,
            feature_grouping=args.feature_grouping,
            gated_residuals=args.gated_residuals,
            multi_layer_decoder=args.multi_layer_decoder,
            decoder_layer_indices=(
                [int(x) for x in args.decoder_layer_indices.split(",")]
                if args.decoder_layer_indices else None
            ),
            max_features=args.max_features,
            n_feature_subsets=args.n_feature_subsets,
        )
    if args.model == "seldon":
        key = args.api_key or os.environ.get("NEURALK_API_KEY")
        if not key:
            raise ValueError(
                "Seldon requires an API key: pass --api_key or set the NEURALK_API_KEY env var."
            )
        return SeldonEvalModel(api_key=key)
    raise ValueError(f"Unknown model: {args.model!r}")
