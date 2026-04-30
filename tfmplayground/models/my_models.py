"""Modified NanoTabPFN with opt-in Priority 1 architectural improvements.

All flags default to False, reproducing baseline NanoTabPFN behaviour exactly.
Each modification can be enabled independently or in combination.

Modifications implemented here:
    Mod 2.11 — target_encoder_use_embedding: Replace TargetEncoder's nn.Linear
        with nn.Embedding, giving each class a free vector and test rows an
        explicit learnable "unknown / predict-me" token.
    Mod 2.1  — target_aware: Add a learnable class embedding to every feature
        cell of every training row before any transformer layer runs, giving
        feature attention immediate access to class-conditional patterns.
    Mod 2.3  — random_perturbations: Add a per-column random perturbation
        r_j = W @ p_j (p_j ~ N(0,1)) to break feature-column symmetry.
        At inference, averaged over n_perturbation_samples draws (handled in
        the classifier wrapper, not here).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import (
    Decoder,
    FeatureEncoder,
    TransformerEncoderLayer,
)


class TargetEncoder(nn.Module):
    """Encodes the target column.

    Baseline (use_embedding=False): pads test rows with mean(y_train) and
    applies a shared nn.Linear(1, d) to every scalar — same as the original.

    Mod 2.11 (use_embedding=True): uses nn.Embedding(num_outputs+1, d).
    Training rows look up their true class index; test rows use a dedicated
    learnable unknown token at index num_outputs.
    """

    def __init__(self, embedding_size: int, num_outputs: int = 1, use_embedding: bool = False):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(num_outputs + 1, embedding_size)
            self._unknown_idx = num_outputs
        else:
            self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Args:
            y_train: (B, n_train, 1) float tensor of integer class indices.
            num_rows: n_train + n_test.
        Returns:
            (B, num_rows, 1, embedding_size)
        """
        if self.use_embedding:
            B, n_train = y_train.shape[0], y_train.shape[1]
            n_test = num_rows - n_train
            y_idx = y_train[:, :, 0].long()  # (B, n_train)
            unknown = torch.full(
                (B, n_test), self._unknown_idx, device=y_train.device, dtype=torch.long
            )
            indices = torch.cat([y_idx, unknown], dim=1)  # (B, num_rows)
            return self.embedding(indices).unsqueeze(2)    # (B, num_rows, 1, d)
        else:
            mean = torch.mean(y_train, axis=1, keepdim=True)
            padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
            y = torch.cat([y_train, padding], dim=1)
            y = y.unsqueeze(-1)
            return self.linear_layer(y)


class MyNanoTabPFNModel(nn.Module):
    """NanoTabPFN with opt-in Priority 1 modifications.

    Constructor flags (all False by default → identical to baseline):
        target_encoder_use_embedding: Mod 2.11
        target_aware:                 Mod 2.1
        random_perturbations:         Mod 2.3
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        target_encoder_use_embedding: bool = False,
        target_aware: bool = False,
        random_perturbations: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.target_encoder_use_embedding = target_encoder_use_embedding
        self.target_aware = target_aware
        self.random_perturbations = random_perturbations

        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(
            embedding_size,
            num_outputs=num_outputs,
            use_embedding=target_encoder_use_embedding,
        )

        # Mod 2.1: one embedding per class, added to feature cells of train rows
        if target_aware:
            self.early_class_embedding = nn.Embedding(num_outputs, embedding_size)

        # Mod 2.3: learned projection W of shape (d, k') maps random column
        # noise p_j ~ N(0,1)^k' to a d-dimensional perturbation r_j = W @ p_j
        if random_perturbations:
            self.k_prime = embedding_size // 4
            self.perturbation_proj = nn.Linear(self.k_prime, embedding_size, bias=False)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            for _ in range(num_layers)
        ])
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Same two-convention dispatch as NanoTabPFNModel.forward."""
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), train_test_split_index=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        num_mem_chunks: int = 1,
    ) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)  # (B, n_train, 1)

        # Step 1: FeatureEncoder — normalize, clip, linear embed → (B, n, m, d)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]

        # Step 2 (Mod 2.3): random attribute perturbations.
        # p_j ~ N(0,1)^k' is re-sampled every forward pass.  At inference the
        # classifier wrapper averages over n_perturbation_samples draws.
        if self.random_perturbations:
            m = x_src.shape[2]
            p = torch.randn(m, self.k_prime, device=x_src.device, dtype=x_src.dtype)
            r = self.perturbation_proj(p)          # (m, d)
            x_src = x_src + r[None, None]          # broadcast (1, 1, m, d) over (B, n, m, d)

        # Step 3 (Mod 2.1): target-aware early embedding.
        # Add class embedding to every feature cell of every training row so
        # that feature attention sees class-conditional patterns from layer 1.
        if self.target_aware:
            y_train_idx = y_src[:, :, 0].long()    # (B, n_train)
            class_emb = self.early_class_embedding(y_train_idx)  # (B, n_train, d)
            # unsqueeze(2) broadcasts over the m feature columns
            train_feat = x_src[:, :train_test_split_index] + class_emb.unsqueeze(2)
            x_src = torch.cat([train_feat, x_src[:, train_test_split_index:]], dim=1)

        # Step 4: TargetEncoder (Mod 2.11 when use_embedding=True) → (B, n, 1, d)
        y_src = self.target_encoder(y_src, num_rows)

        # Step 5: concatenate feature and target embeddings → (B, n, m+1, d)
        src = torch.cat([x_src, y_src], dim=2)

        # Step 6: transformer layers (num_mem_chunks bug fixed in nanotabpfn.py)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index,
                        num_mem_chunks=num_mem_chunks)

        # Step 7: extract test-row target-column embeddings and decode
        output = src[:, train_test_split_index:, -1, :]  # (B, n_test, d)
        return self.decoder(output)                       # (B, n_test, num_outputs)


def init_my_model_from_state_dict_file(file_path: str) -> MyNanoTabPFNModel:
    """Loads a MyNanoTabPFNModel checkpoint saved by the training loop."""
    state_dict = torch.load(file_path, map_location="cpu")
    arch = state_dict["architecture"]
    model = MyNanoTabPFNModel(
        num_attention_heads=arch["num_attention_heads"],
        embedding_size=arch["embedding_size"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
        target_encoder_use_embedding=arch.get("target_encoder_use_embedding", False),
        target_aware=arch.get("target_aware", False),
        random_perturbations=arch.get("random_perturbations", False),
    )
    model.load_state_dict(state_dict["model"])
    return model
