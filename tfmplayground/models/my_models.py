"""Modified NanoTabPFN with opt-in architectural improvements.

All flags default to False/None, reproducing baseline behaviour exactly.

Priority 1:  target_encoder_use_embedding (Mod 2.11)
             target_aware                 (Mod 2.1)
             random_perturbations         (Mod 2.3)

Priority 2:  prenorm          (Mod 2.5) — pre-norm LN + final LN
             cls_compression  (Mod 2.4) — [CLS] row compression
             (Mod 2.6 lives in interface.py — inference-time only)

Priority 3:  feature_grouping      (Mod 2.2) — circular grouped FeatureEncoder
             gated_residuals       (Mod 2.8) — softplus-gated residual connections
             multi_layer_decoder   (Mod 2.9) — concatenate intermediate layer outputs
             (Mod 2.10 lives in interface.py — inference-time only)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention

from tfmplayground.models.nanotabpfn import Decoder, FeatureEncoder, memory_chunking


# ---------------------------------------------------------------------------
# TargetEncoder (Mod 2.11)
# ---------------------------------------------------------------------------

class TargetEncoder(nn.Module):
    """Baseline: nn.Linear(1, d) + mean-padding for test rows.
    Mod 2.11: nn.Embedding(num_outputs+1, d) with a learnable unknown token
    at index num_outputs for test rows.
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
            num_rows: total rows n = n_train + n_test.
        Returns: (B, num_rows, 1, embedding_size)
        """
        if self.use_embedding:
            B, n_train = y_train.shape[0], y_train.shape[1]
            n_test = num_rows - n_train
            y_idx = y_train[:, :, 0].long()
            unknown = torch.full(
                (B, n_test), self._unknown_idx, device=y_train.device, dtype=torch.long
            )
            indices = torch.cat([y_idx, unknown], dim=1)
            return self.embedding(indices).unsqueeze(2)
        else:
            mean = torch.mean(y_train, axis=1, keepdim=True)
            padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
            y = torch.cat([y_train, padding], dim=1)
            return self.linear_layer(y.unsqueeze(-1))


# ---------------------------------------------------------------------------
# GroupedFeatureEncoder (Mod 2.2)
# ---------------------------------------------------------------------------

class GroupedFeatureEncoder(nn.Module):
    """Replaces the per-scalar nn.Linear(1, d) with a circular grouped encoding.

    For each column j the input is [x_j, x_{(j+1)%m}, x_{(j+3)%m}], forming
    a 3-element vector that is projected by a shared nn.Linear(3, d).

    Note: at m=3 the grouping is degenerate (every group contains all three
    columns) so this modification is only meaningful at m ≥ 7.  It is
    included for completeness as specified by the architecture plan.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(3, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """x: (B, n, m) → (B, n, m, d)"""
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdim=True)
        std  = torch.std(x[:, :train_test_split_index],  dim=1, keepdim=True) + 1e-8
        x = torch.clip((x - mean) / std, min=-100, max=100).squeeze(-1)  # (B, n, m)
        m = x.shape[2]
        j  = torch.arange(m, device=x.device)
        j1 = (j + 1) % m
        j3 = (j + 3) % m
        grouped = torch.stack([x[..., j], x[..., j1], x[..., j3]], dim=-1)  # (B, n, m, 3)
        return self.linear_layer(grouped)                                    # (B, n, m, d)


# ---------------------------------------------------------------------------
# TransformerEncoderLayer (Mod 2.5 prenorm + Mod 2.4 Stage-1 CLS + Mod 2.8 gates)
# ---------------------------------------------------------------------------

_GATE_INIT = 0.5413  # softplus(0.5413) ≈ 1.0  → baseline residual scale at init


class TransformerEncoderLayer(nn.Module):
    """Bi-attention transformer layer with three opt-in extensions.

    prenorm (Mod 2.5): LayerNorm before each sublayer instead of after.
    n_cls_tokens (Mod 2.4 Stage 1): exclude the last n_cls_tokens columns
        from datapoint attention.
    gated_residuals (Mod 2.8): each residual branch is scaled by
        softplus(gate) where gate is a per-sublayer learnable scalar,
        initialised so the scale is ≈1.0 at the start of training.

    When all extensions are off the output is identical to the original
    nanotabpfn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        prenorm: bool = False,
        n_cls_tokens: int = 0,
        gated_residuals: bool = False,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.n_cls_tokens = n_cls_tokens
        self.gated_residuals = gated_residuals

        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

        if gated_residuals:
            self.gate_feat = nn.Parameter(torch.tensor(_GATE_INIT))
            self.gate_dp   = nn.Parameter(torch.tensor(_GATE_INIT))
            self.gate_mlp  = nn.Parameter(torch.tensor(_GATE_INIT))

    def forward(self, src: torch.Tensor, train_test_split_index: int,
                num_mem_chunks: int = 1) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # ── Feature attention: over ALL cols ──────────────────────────────────
        src_flat = src.reshape(batch_size * rows_size, col_size, embedding_size)
        gate_feat = F.softplus(self.gate_feat) if self.gated_residuals else 1.0

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            n = self.norm1(x) if self.prenorm else x
            return gate_feat * self.self_attention_between_features(n, n, n)[0] + x

        src_flat = feature_attention(src_flat)
        src = src_flat.reshape(batch_size, rows_size, col_size, embedding_size)
        if not self.prenorm:
            src = self.norm1(src)

        # ── Datapoint attention: content cols only (CLS excluded) ─────────────
        n_content = col_size - self.n_cls_tokens
        src_content = src[:, :, :n_content, :]
        src_cls     = src[:, :, n_content:, :]

        src_content = src_content.transpose(1, 2).reshape(
            batch_size * n_content, rows_size, embedding_size
        )
        gate_dp = F.softplus(self.gate_dp) if self.gated_residuals else 1.0

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            n = self.norm2(x) if self.prenorm else x
            x_left = self.self_attention_between_datapoints(
                n[:, :train_test_split_index],
                n[:, :train_test_split_index],
                n[:, :train_test_split_index],
            )[0]
            x_right = self.self_attention_between_datapoints(
                n[:, train_test_split_index:],
                n[:, :train_test_split_index],
                n[:, :train_test_split_index],
            )[0]
            return gate_dp * torch.cat([x_left, x_right], dim=1) + x

        src_content = datapoint_attention(src_content)
        src_content = src_content.reshape(
            batch_size, n_content, rows_size, embedding_size
        ).transpose(2, 1)
        if not self.prenorm:
            src_content = self.norm2(src_content)

        src = torch.cat([src_content, src_cls], dim=2)

        # ── Cell-wise MLP: over ALL cols ──────────────────────────────────────
        src = src.reshape(-1, embedding_size)
        gate_mlp = F.softplus(self.gate_mlp) if self.gated_residuals else 1.0

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            n = self.norm3(x) if self.prenorm else x
            return gate_mlp * self.linear2(F.gelu(self.linear1(n))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        if not self.prenorm:
            src = self.norm3(src)
        return src


# ---------------------------------------------------------------------------
# Stage2TransformerLayer (Mod 2.4 + Mod 2.5 + Mod 2.8)
# ---------------------------------------------------------------------------

class Stage2TransformerLayer(nn.Module):
    """Plain self-attention on compressed row embeddings (B, n, K*d).

    Applies the PFN mask: train→train and test→train only.
    Supports prenorm (Mod 2.5) and gated residuals (Mod 2.8).
    """

    def __init__(
        self,
        row_dim: int,
        nhead: int,
        mlp_hidden_size: int,
        prenorm: bool = False,
        gated_residuals: bool = False,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.gated_residuals = gated_residuals
        self.attn = MultiheadAttention(
            row_dim, nhead, batch_first=True, device=device, dtype=dtype
        )
        self.linear1 = Linear(row_dim, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, row_dim,  device=device, dtype=dtype)
        self.norm1 = LayerNorm(row_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(row_dim, eps=layer_norm_eps, device=device, dtype=dtype)

        if gated_residuals:
            self.gate_attn = nn.Parameter(torch.tensor(_GATE_INIT))
            self.gate_mlp  = nn.Parameter(torch.tensor(_GATE_INIT))

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x_n = self.norm1(x) if self.prenorm else x
        attn_out = torch.cat([
            self.attn(x_n[:, :train_test_split_index],
                      x_n[:, :train_test_split_index],
                      x_n[:, :train_test_split_index])[0],
            self.attn(x_n[:, train_test_split_index:],
                      x_n[:, :train_test_split_index],
                      x_n[:, :train_test_split_index])[0],
        ], dim=1)
        gate_attn = F.softplus(self.gate_attn) if self.gated_residuals else 1.0
        x = x + gate_attn * attn_out
        if not self.prenorm:
            x = self.norm1(x)

        x_n = self.norm2(x) if self.prenorm else x
        gate_mlp = F.softplus(self.gate_mlp) if self.gated_residuals else 1.0
        x = x + gate_mlp * self.linear2(F.gelu(self.linear1(x_n)))
        if not self.prenorm:
            x = self.norm2(x)
        return x


# ---------------------------------------------------------------------------
# MyNanoTabPFNModel
# ---------------------------------------------------------------------------

class MyNanoTabPFNModel(nn.Module):
    """NanoTabPFN with opt-in P1/P2/P3 modifications.

    All flags default to False/0/None — reproduces baseline exactly.

    P1: target_encoder_use_embedding, target_aware, random_perturbations
    P2: prenorm, cls_compression, n_cls_tokens, n_stage1_layers,
        n_stage2_layers, icl_target_embedding
    P3: feature_grouping, gated_residuals, multi_layer_decoder,
        decoder_layer_indices
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        # P1
        target_encoder_use_embedding: bool = False,
        target_aware: bool = False,
        random_perturbations: bool = False,
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
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        # P1
        self.target_encoder_use_embedding = target_encoder_use_embedding
        self.target_aware = target_aware
        self.random_perturbations = random_perturbations
        # P2
        self.prenorm = prenorm
        self.cls_compression = cls_compression
        self.n_cls_tokens = n_cls_tokens
        self.n_stage1_layers = n_stage1_layers
        self.n_stage2_layers = n_stage2_layers
        self.icl_target_embedding = icl_target_embedding
        # P3
        self.feature_grouping = feature_grouping
        self.gated_residuals = gated_residuals
        self.multi_layer_decoder = multi_layer_decoder

        # ── Feature encoder ──
        if feature_grouping:
            self.feature_encoder = GroupedFeatureEncoder(embedding_size)
        else:
            self.feature_encoder = FeatureEncoder(embedding_size)

        # ── Target encoder ──
        self.target_encoder = TargetEncoder(
            embedding_size, num_outputs=num_outputs,
            use_embedding=target_encoder_use_embedding,
        )

        # ── P1 optional modules ──
        if target_aware:
            self.early_class_embedding = nn.Embedding(num_outputs, embedding_size)
        if random_perturbations:
            self.k_prime = embedding_size // 4
            self.perturbation_proj = nn.Linear(self.k_prime, embedding_size, bias=False)

        # ── Transformer layers ──
        _layer_kw = dict(prenorm=prenorm, gated_residuals=gated_residuals)

        if cls_compression:
            self.cls_tokens = nn.Parameter(
                torch.empty(n_cls_tokens, embedding_size).normal_(std=0.02)
            )
            self.stage1_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size,
                    n_cls_tokens=n_cls_tokens, **_layer_kw,
                )
                for _ in range(n_stage1_layers)
            ])
            base_dim = n_cls_tokens * embedding_size
            if icl_target_embedding:
                self.icl_class_embedding = nn.Embedding(num_outputs, base_dim)
            self.stage2_layers = nn.ModuleList([
                Stage2TransformerLayer(
                    base_dim, num_attention_heads, mlp_hidden_size, **_layer_kw,
                )
                for _ in range(n_stage2_layers)
            ])
            n_extractable = n_stage2_layers
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size, **_layer_kw,
                )
                for _ in range(num_layers)
            ])
            base_dim = embedding_size
            n_extractable = num_layers

        # ── Mod 2.9: multi-layer decoder indices ──
        if multi_layer_decoder:
            if decoder_layer_indices is None:
                decoder_layer_indices = list(range(1, n_extractable + 1))
            decoder_input_dim = len(decoder_layer_indices) * base_dim
        else:
            decoder_layer_indices = None
            decoder_input_dim = base_dim
        self.decoder_layer_indices = decoder_layer_indices

        # ── Final LayerNorm (prenorm only; always over base_dim) ──
        if prenorm:
            self.final_layer_norm = nn.LayerNorm(base_dim)

        self.decoder = Decoder(decoder_input_dim, mlp_hidden_size, num_outputs)

    # ── Forward dispatch ─────────────────────────────────────────────────────

    def forward(self, *args, **kwargs) -> torch.Tensor:
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
            y_src = y_src.unsqueeze(-1)

        B = x_src.shape[0]
        y_src_indices = y_src

        # Step 1: FeatureEncoder (Mod 2.2 uses GroupedFeatureEncoder when active)
        x_src = self.feature_encoder(x_src, train_test_split_index)  # (B, n, m, d)
        num_rows = x_src.shape[1]

        # Step 2 (Mod 2.3): random attribute perturbations
        if self.random_perturbations:
            m = x_src.shape[2]
            p = torch.randn(m, self.k_prime, device=x_src.device, dtype=x_src.dtype)
            r = self.perturbation_proj(p)
            x_src = x_src + r[None, None]

        # Step 3 (Mod 2.1): target-aware early embedding on training rows
        if self.target_aware:
            y_idx = y_src_indices[:, :, 0].long()
            class_emb = self.early_class_embedding(y_idx)              # (B, n_train, d)
            train_feat = x_src[:, :train_test_split_index] + class_emb.unsqueeze(2)
            x_src = torch.cat([train_feat, x_src[:, train_test_split_index:]], dim=1)

        # Step 4: TargetEncoder
        y_emb = self.target_encoder(y_src_indices, num_rows)          # (B, n, 1, d)

        # Step 5: concatenate features + target → (B, n, m+1, d)
        src = torch.cat([x_src, y_emb], dim=2)

        # ── CLS-compression path (Mod 2.4) ───────────────────────────────────
        if self.cls_compression:
            n_content_cols = src.shape[2]
            cls = self.cls_tokens[None, None].expand(B, num_rows, -1, -1)
            src = torch.cat([src, cls], dim=2)

            for block in self.stage1_layers:
                src = block(src, train_test_split_index=train_test_split_index,
                            num_mem_chunks=num_mem_chunks)

            cls_out = src[:, :, n_content_cols:, :]
            row_emb = cls_out.reshape(B, num_rows,
                                      self.n_cls_tokens * self.embedding_size)

            if self.icl_target_embedding:
                y_idx = y_src_indices[:, :, 0].long()
                icl_emb = self.icl_class_embedding(y_idx)
                train_row = row_emb[:, :train_test_split_index] + icl_emb
                row_emb = torch.cat([train_row, row_emb[:, train_test_split_index:]], dim=1)

            # Stage 2 — with optional multi-layer extraction (Mod 2.9)
            if self.multi_layer_decoder:
                extracted = []
                for i, layer in enumerate(self.stage2_layers):
                    row_emb = layer(row_emb, train_test_split_index=train_test_split_index)
                    if (i + 1) in self.decoder_layer_indices:
                        emb = row_emb[:, train_test_split_index:, :]
                        if self.prenorm:
                            emb = self.final_layer_norm(emb)
                        extracted.append(emb)
                output = torch.cat(extracted, dim=-1)
            else:
                for layer in self.stage2_layers:
                    row_emb = layer(row_emb, train_test_split_index=train_test_split_index)
                if self.prenorm:
                    row_emb = self.final_layer_norm(row_emb)
                output = row_emb[:, train_test_split_index:, :]

        # ── Standard path ────────────────────────────────────────────────────
        else:
            # Mod 2.9: collect embeddings from intermediate layers
            if self.multi_layer_decoder:
                extracted = []
                for i, block in enumerate(self.transformer_blocks):
                    src = block(src, train_test_split_index=train_test_split_index,
                                num_mem_chunks=num_mem_chunks)
                    if (i + 1) in self.decoder_layer_indices:
                        emb = src[:, train_test_split_index:, -1, :]
                        if self.prenorm:
                            emb = self.final_layer_norm(emb)
                        extracted.append(emb)
                output = torch.cat(extracted, dim=-1)
            else:
                for block in self.transformer_blocks:
                    src = block(src, train_test_split_index=train_test_split_index,
                                num_mem_chunks=num_mem_chunks)
                if self.prenorm:
                    src = self.final_layer_norm(src)
                output = src[:, train_test_split_index:, -1, :]

        return self.decoder(output)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_BOOL_FLAGS = (
    "target_aware", "random_perturbations", "target_encoder_use_embedding",
    "prenorm", "cls_compression", "icl_target_embedding",
    "feature_grouping", "gated_residuals", "multi_layer_decoder",
)
_INT_FLAGS  = ("n_cls_tokens", "n_stage1_layers", "n_stage2_layers")
_LIST_FLAGS = ("decoder_layer_indices",)


def init_my_model_from_state_dict_file(file_path: str) -> MyNanoTabPFNModel:
    """Loads a MyNanoTabPFNModel from a checkpoint produced by the training loop."""
    state_dict = torch.load(file_path, map_location="cpu")
    arch = state_dict["architecture"]
    model = MyNanoTabPFNModel(
        num_attention_heads=arch["num_attention_heads"],
        embedding_size=arch["embedding_size"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
        # P1
        target_encoder_use_embedding=arch.get("target_encoder_use_embedding", False),
        target_aware=arch.get("target_aware", False),
        random_perturbations=arch.get("random_perturbations", False),
        # P2
        prenorm=arch.get("prenorm", False),
        cls_compression=arch.get("cls_compression", False),
        n_cls_tokens=arch.get("n_cls_tokens", 2),
        n_stage1_layers=arch.get("n_stage1_layers", 3),
        n_stage2_layers=arch.get("n_stage2_layers", 3),
        icl_target_embedding=arch.get("icl_target_embedding", False),
        # P3
        feature_grouping=arch.get("feature_grouping", False),
        gated_residuals=arch.get("gated_residuals", False),
        multi_layer_decoder=arch.get("multi_layer_decoder", False),
        decoder_layer_indices=arch.get("decoder_layer_indices", None),
    )
    model.load_state_dict(state_dict["model"])
    return model
