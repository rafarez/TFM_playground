import math
import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device

_MY_MODEL_FLAGS = (
    "target_aware", "random_perturbations", "target_encoder_use_embedding",
    "prenorm", "cls_compression", "icl_target_embedding",
)


def init_model_from_state_dict_file(file_path):
    """Loads a checkpoint, auto-detecting model type from the architecture dict.

    Any P1/P2 modification flag present in the checkpoint means the file was
    produced by MyNanoTabPFNModel; otherwise the baseline NanoTabPFNModel is
    used.  Backward-compatible with all existing checkpoints.
    """
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    arch = state_dict["architecture"]
    if any(f in arch for f in _MY_MODEL_FLAGS):
        from tfmplayground.models.my_models import MyNanoTabPFNModel
        model = MyNanoTabPFNModel(
            num_attention_heads=arch["num_attention_heads"],
            embedding_size=arch["embedding_size"],
            mlp_hidden_size=arch["mlp_hidden_size"],
            num_layers=arch["num_layers"],
            num_outputs=arch["num_outputs"],
            target_encoder_use_embedding=arch.get("target_encoder_use_embedding", False),
            target_aware=arch.get("target_aware", False),
            random_perturbations=arch.get("random_perturbations", False),
            prenorm=arch.get("prenorm", False),
            cls_compression=arch.get("cls_compression", False),
            n_cls_tokens=arch.get("n_cls_tokens", 2),
            n_stage1_layers=arch.get("n_stage1_layers", 3),
            n_stage2_layers=arch.get("n_stage2_layers", 3),
            icl_target_embedding=arch.get("icl_target_embedding", False),
        )
    else:
        model = NanoTabPFNModel(
            num_attention_heads=arch["num_attention_heads"],
            embedding_size=arch["embedding_size"],
            mlp_hidden_size=arch["mlp_hidden_size"],
            num_layers=arch["num_layers"],
            num_outputs=arch["num_outputs"],
        )
    model.load_state_dict(state_dict["model"])
    return model


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),  # to apply pd.to_numeric of pandas
            ("to_numeric", FunctionTransformer(to_numeric)),  # in case numeric columns are stored as strings
            (
                "imputer",
                SimpleImputer(strategy="mean", add_indicator=True),
            ),  # median might be better because of outliers
        ]
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_mask), ("cat", cat_transformer, cat_mask)]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
        n_perturbation_samples: int = 1,
        max_features: int | None = None,
        n_feature_subsets: int | None = None,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        self.n_perturbation_samples = n_perturbation_samples
        self.max_features = max_features
        self.n_feature_subsets = n_feature_subsets

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """stores X_train and y_train for later use, also computes the highest class number occuring in num_classes"""
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predit_proba and picks the class with the highest probability for each datapoint"""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def _run_forward_pass(
        self, x_train_pre: np.ndarray, x_test_pre: np.ndarray
    ) -> np.ndarray:
        """Single forward pass on already-preprocessed feature arrays."""
        x = np.concatenate((x_train_pre, x_test_pre))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model(
                (x, y), train_test_split_index=len(x_train_pre),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)
            out = out[:, : self.num_classes]
            return F.softmax(out, dim=1).to("cpu").numpy()

    def _predict_proba_once(self, X_test: np.ndarray) -> np.ndarray:
        """Single forward pass (no perturbation averaging).  Preprocesses X_test."""
        return self._run_forward_pass(
            self.X_train, self.feature_preprocessor.transform(X_test)
        )

    def _forward_averaged(
        self, x_train_pre: np.ndarray, x_test_pre: np.ndarray
    ) -> np.ndarray:
        """Forward pass averaged over n_perturbation_samples draws (Mod 2.3)."""
        if self.n_perturbation_samples <= 1:
            return self._run_forward_pass(x_train_pre, x_test_pre)
        total = sum(
            self._run_forward_pass(x_train_pre, x_test_pre)
            for _ in range(self.n_perturbation_samples)
        )
        return total / self.n_perturbation_samples

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Returns class probabilities for X_test.

        Supports two optional inference-time strategies that can be combined:
          n_perturbation_samples > 1  (Mod 2.3): average over random-perturbation draws.
          max_features               (Mod 2.6): feature subspace bagging — run on
              n_feature_subsets random feature subsets of size max_features and
              average the predicted probabilities.
        """
        x_test_pre = self.feature_preprocessor.transform(X_test)
        d = self.X_train.shape[1]

        if self.max_features is None or d <= self.max_features:
            return self._forward_averaged(self.X_train, x_test_pre)

        # Mod 2.6: feature subspace bagging
        n_subsets = self.n_feature_subsets or math.ceil(d / self.max_features)
        rng = np.random.default_rng()
        total = None
        for _ in range(n_subsets):
            idx = rng.choice(d, min(self.max_features, d), replace=False)
            p = self._forward_averaged(self.X_train[:, idx], x_test_pre[:, idx])
            total = p if total is None else total + p
        return total / n_subsets


class NanoTabPFNRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
                )
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor), train_test_split_index=len(self.X_train), num_mem_chunks=self.num_mem_chunks
            ).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
