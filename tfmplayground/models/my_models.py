"""Modified NanoTabPFN with opt-in architectural improvements.

All flags default to False/None, reproducing baseline behaviour exactly.

Priority 1 (implemented previously):
    target_encoder_use_embedding (Mod 2.11)
    target_aware                 (Mod 2.1)
    random_perturbations         (Mod 2.3)

Priority 2 (this file):
    prenorm          (Mod 2.5) — pre-norm LayerNorm in all sublayers + final LN
    cls_compression  (Mod 2.4) — [CLS] row compression between Stage 1 and Stage 2
    (inference-time) max_features / n_feature_subsets (Mod 2.6) live in interface.py
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
    Mod 2.11: nn.Embedding(num_outputs+1, d) with a dedicated unknown token
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
        Returns:
            (B, num_rows, 1, embedding_size)
        """
        if self.use_embedding:
            B, n_train = y_train.shape[0], y_train.shape[1]
            n_test = num_rows - n_train
            y_idx = y_train[:, :, 0].long()
            unknown = torch.full(
                (B, n_test), self._unknown_idx, device=y_train.device, dtype=torch.long
            )
            indices = torch.cat([y_idx, unknown], dim=1)   # (B, num_rows)
            return self.embedding(indices).unsqueeze(2)    # (B, num_rows, 1, d)
        else:
            mean = torch.mean(y_train, axis=1, keepdim=True)
            padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
            y = torch.cat([y_train, padding], dim=1)
            y = y.unsqueeze(-1)
            return self.linear_layer(y)


# ---------------------------------------------------------------------------
# TransformerEncoderLayer (Mod 2.5 + Mod 2.4 Stage 1)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Bi-attention transformer layer.

    Supports two opt-in extensions that are off by default:
      prenorm     (Mod 2.5): apply LayerNorm before each sublayer instead of after.
      n_cls_tokens (Mod 2.4 Stage 1): exclude the last n_cls_tokens columns
          from datapoint attention (CLS tokens participate only in feature
          attention and the MLP, following the TabICLv2 design).

    When prenorm=False and n_cls_tokens=0 the output is numerically identical
    to the original nanotabpfn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        prenorm: bool = False,
        n_cls_tokens: int = 0,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.n_cls_tokens = n_cls_tokens

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

    def forward(self, src: torch.Tensor, train_test_split_index: int,
                num_mem_chunks: int = 1) -> torch.Tensor:
        """
        src: (B, rows, cols, d)   cols = m+1 [+K when n_cls_tokens > 0]
        """
        batch_size, rows_size, col_size, embedding_size = src.shape

        # ── Feature attention: over ALL cols (content + CLS if present) ──
        src_flat = src.reshape(batch_size * rows_size, col_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            n = self.norm1(x) if self.prenorm else x
            return self.self_attention_between_features(n, n, n)[0] + x

        src_flat = feature_attention(src_flat)
        src = src_flat.reshape(batch_size, rows_size, col_size, embedding_size)
        if not self.prenorm:
            src = self.norm1(src)

        # ── Datapoint attention: content cols only (exclude CLS) ──
        n_content = col_size - self.n_cls_tokens   # = m+1 when CLS present
        src_content = src[:, :, :n_content, :]     # (B, n, m+1, d)
        src_cls     = src[:, :, n_content:, :]     # (B, n, K, d) or empty

        src_content = src_content.transpose(1, 2)                          # (B, m+1, n, d)
        src_content = src_content.reshape(batch_size * n_content, rows_size, embedding_size)

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
            return torch.cat([x_left, x_right], dim=1) + x

        src_content = datapoint_attention(src_content)
        src_content = src_content.reshape(batch_size, n_content, rows_size, embedding_size)
        src_content = src_content.transpose(2, 1)      # (B, n, m+1, d)
        if not self.prenorm:
            src_content = self.norm2(src_content)

        src = torch.cat([src_content, src_cls], dim=2)  # reattach CLS (no-op when K=0)

        # ── Cell-wise MLP: over ALL cols ──
        src = src.reshape(-1, embedding_size)

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            n = self.norm3(x) if self.prenorm else x
            return self.linear2(F.gelu(self.linear1(n))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        if not self.prenorm:
            src = self.norm3(src)
        return src


# ---------------------------------------------------------------------------
# Stage2TransformerLayer (Mod 2.4)
# ---------------------------------------------------------------------------

class Stage2TransformerLayer(nn.Module):
    """Plain self-attention layer for Stage 2 (ICL on compressed row embeddings).

    Input/output: (B, n, row_dim) where row_dim = K * embedding_size.
    Applies the PFN causal mask: train rows attend to each other; test rows
    attend only to train rows (never to other test rows).
    Supports pre-norm (Mod 2.5).
    """

    def __init__(
        self,
        row_dim: int,
        nhead: int,
        mlp_hidden_size: int,
        prenorm: bool = False,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.attn = MultiheadAttention(
            row_dim, nhead, batch_first=True, device=device, dtype=dtype
        )
        self.linear1 = Linear(row_dim, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, row_dim, device=device, dtype=dtype)
        self.norm1 = LayerNorm(row_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(row_dim, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """x: (B, n, row_dim)"""
        # Self-attention with PFN mask
        x_n = self.norm1(x) if self.prenorm else x
        attn_left  = self.attn(x_n[:, :train_test_split_index],
                               x_n[:, :train_test_split_index],
                               x_n[:, :train_test_split_index])[0]
        attn_right = self.attn(x_n[:, train_test_split_index:],
                               x_n[:, :train_test_split_index],
                               x_n[:, :train_test_split_index])[0]
        x = x + torch.cat([attn_left, attn_right], dim=1)
        if not self.prenorm:
            x = self.norm1(x)

        # MLP
        x_n = self.norm2(x) if self.prenorm else x
        x = x + self.linear2(F.gelu(self.linear1(x_n)))
        if not self.prenorm:
            x = self.norm2(x)
        return x


# ---------------------------------------------------------------------------
# MyNanoTabPFNModel
# ---------------------------------------------------------------------------

class MyNanoTabPFNModel(nn.Module):
    """NanoTabPFN with opt-in Priority 1 and Priority 2 modifications.

    All flags default to False/0, reproducing baseline behaviour exactly.

    P1 flags: target_encoder_use_embedding, target_aware, random_perturbations
    P2 flags: prenorm, cls_compression, n_cls_tokens, n_stage1_layers,
              n_stage2_layers, icl_target_embedding
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

        # ── Encoders ──
        self.feature_encoder = FeatureEncoder(embedding_size)
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

        # ── Transformer layers and decoder ──
        if cls_compression:
            # CLS tokens shared across all rows; expanded in _forward
            self.cls_tokens = nn.Parameter(
                torch.empty(n_cls_tokens, embedding_size).normal_(std=0.02)
            )
            # Stage 1: bi-attention with CLS exclusion from datapoint attention
            self.stage1_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size,
                    prenorm=prenorm, n_cls_tokens=n_cls_tokens,
                )
                for _ in range(n_stage1_layers)
            ])
            row_dim = n_cls_tokens * embedding_size
            # Optional second label injection into compressed row embeddings
            if icl_target_embedding:
                self.icl_class_embedding = nn.Embedding(num_outputs, row_dim)
            # Stage 2: plain self-attention on row embeddings
            self.stage2_layers = nn.ModuleList([
                Stage2TransformerLayer(
                    row_dim, num_attention_heads, mlp_hidden_size, prenorm=prenorm,
                )
                for _ in range(n_stage2_layers)
            ])
            decoder_input_dim = row_dim
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size, prenorm=prenorm,
                )
                for _ in range(num_layers)
            ])
            decoder_input_dim = embedding_size

        # Final LayerNorm required with pre-norm to keep the residual stream bounded
        if prenorm:
            self.final_layer_norm = nn.LayerNorm(decoder_input_dim)

        self.decoder = Decoder(decoder_input_dim, mlp_hidden_size, num_outputs)

    # ── Forward dispatch (identical to NanoTabPFNModel.forward) ──────────────

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
            y_src = y_src.unsqueeze(-1)          # (B, n_train, 1)

        B = x_src.shape[0]
        y_src_indices = y_src                    # saved for ICL embedding

        # Step 1: FeatureEncoder → (B, n, m, d)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]

        # Step 2 (Mod 2.3): random attribute perturbations
        if self.random_perturbations:
            m = x_src.shape[2]
            p = torch.randn(m, self.k_prime, device=x_src.device, dtype=x_src.dtype)
            r = self.perturbation_proj(p)        # (m, d)
            x_src = x_src + r[None, None]        # broadcast (1, 1, m, d)

        # Step 3 (Mod 2.1): target-aware early embedding on training rows
        if self.target_aware:
            y_train_idx = y_src_indices[:, :, 0].long()          # (B, n_train)
            class_emb   = self.early_class_embedding(y_train_idx) # (B, n_train, d)
            train_feat  = x_src[:, :train_test_split_index] + class_emb.unsqueeze(2)
            x_src = torch.cat([train_feat, x_src[:, train_test_split_index:]], dim=1)

        # Step 4: TargetEncoder (Mod 2.11 when use_embedding=True) → (B, n, 1, d)
        y_emb = self.target_encoder(y_src_indices, num_rows)

        # Step 5: concatenate → (B, n, m+1, d)
        src = torch.cat([x_src, y_emb], dim=2)

        # ── CLS-compression path (Mod 2.4) ──────────────────────────────────
        if self.cls_compression:
            n_content_cols = src.shape[2]        # = m+1, before CLS appended

            # Append K CLS tokens (shared, expanded across B and n)
            cls = self.cls_tokens[None, None].expand(B, num_rows, -1, -1)  # (B, n, K, d)
            src = torch.cat([src, cls], dim=2)   # (B, n, m+1+K, d)

            # Stage 1: bi-attention with CLS excluded from datapoint attention
            for block in self.stage1_layers:
                src = block(src, train_test_split_index=train_test_split_index,
                            num_mem_chunks=num_mem_chunks)

            # Row compression: extract CLS outputs → (B, n, K*d)
            cls_out = src[:, :, n_content_cols:, :]               # (B, n, K, d)
            row_emb = cls_out.reshape(B, num_rows,
                                      self.n_cls_tokens * self.embedding_size)

            # Optional ICL-stage label injection into compressed row embeddings
            if self.icl_target_embedding:
                y_train_idx = y_src_indices[:, :, 0].long()
                icl_emb = self.icl_class_embedding(y_train_idx)   # (B, n_train, K*d)
                train_row = row_emb[:, :train_test_split_index] + icl_emb
                row_emb = torch.cat([train_row, row_emb[:, train_test_split_index:]], dim=1)

            # Stage 2: plain self-attention on row embeddings
            for layer in self.stage2_layers:
                row_emb = layer(row_emb, train_test_split_index=train_test_split_index)

            if self.prenorm:
                row_emb = self.final_layer_norm(row_emb)

            output = row_emb[:, train_test_split_index:, :]       # (B, n_test, K*d)

        # ── Standard path (no CLS compression) ──────────────────────────────
        else:
            for block in self.transformer_blocks:
                src = block(src, train_test_split_index=train_test_split_index,
                            num_mem_chunks=num_mem_chunks)

            if self.prenorm:
                src = self.final_layer_norm(src)

            output = src[:, train_test_split_index:, -1, :]       # (B, n_test, d)

        return self.decoder(output)                                # (B, n_test, num_outputs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_BOOL_FLAGS = (
    "target_aware", "random_perturbations", "target_encoder_use_embedding",
    "prenorm", "cls_compression", "icl_target_embedding",
)
_INT_FLAGS = ("n_cls_tokens", "n_stage1_layers", "n_stage2_layers")


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
    )
    model.load_state_dict(state_dict["model"])
    return model
