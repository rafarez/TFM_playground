"""CLI entry point for the TFM-Playground benchmark harness.

Examples:
    python evaluate.py --model nanotabpfn --benchmark openml-cc18 --protocol fixed
    python evaluate.py --model nanotabpfn --protocol 5fold --auc_only
    python evaluate.py --model nanotabpfn --checkpoint workdir/my_run.pth
    python evaluate.py --model seldon --api_key nk_live_xxx
"""

from __future__ import annotations

import argparse

from tfmplayground.benchmark.models import build_model
from tfmplayground.benchmark.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classification benchmark harness for TFM-Playground models."
    )
    p.add_argument("--model", choices=["nanotabpfn", "mynanotabpfn", "seldon"],
                   default="nanotabpfn",
                   help="Which model to evaluate (default: nanotabpfn).")
    p.add_argument("--benchmark", default="tabarena",
                   help="Benchmark suite name (default: tabarena).")
    p.add_argument("--task_ids", default=None,
                   help="Comma-separated OpenML task IDs. Overrides --benchmark if set.")
    p.add_argument("--protocol", choices=["fixed", "5fold"], default="fixed",
                   help="Evaluation protocol (default: fixed stratified 80/20 split).")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Test-set fraction for --protocol fixed (default: 0.2).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed controlling both splits (default: 42).")
    p.add_argument("--auc_only", action="store_true",
                   help="Only compute ROC AUC (skip accuracy and log-loss).")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Test-time batch size for predict_proba chunks (default: 128). "
                        "Lower this if you hit CUDA OOM on large test sets.")
    p.add_argument("--output_dir", default="experiments",
                   help="Root directory for experiment folders (default: experiments).")
    p.add_argument("--cache_dir", default=None,
                   help="OpenML cache directory (default: OpenML default).")
    p.add_argument("--max_n_features", type=int, default=5000,
                   help="Skip tasks with more than this many features (default: 5000).")
    p.add_argument("--max_n_samples", type=int, default=10_000,
                   help="Skip tasks with more than this many instances (default: 10000).")

    # nanotabpfn-specific
    p.add_argument("--checkpoint", default=None,
                   help="[nanotabpfn] Path to a local checkpoint .pth file. "
                        "If unset, the released pretrained checkpoint is downloaded.")
    p.add_argument("--device", default=None,
                   help="[nanotabpfn] torch device (e.g. 'cuda', 'cpu').")
    p.add_argument("--num_mem_chunks", type=int, default=8,
                   help="[nanotabpfn] Attention chunking factor (default: 8).")

    # mynanotabpfn — Priority 1 flags
    p.add_argument("--target_aware", action="store_true",
                   help="[mynanotabpfn] Mod 2.1: class embedding added to every "
                        "feature cell of training rows before the transformer layers.")
    p.add_argument("--random_perturbations", action="store_true",
                   help="[mynanotabpfn] Mod 2.3: per-column random perturbations "
                        "to break feature-column symmetry.")
    p.add_argument("--target_encoder_use_embedding", action="store_true",
                   help="[mynanotabpfn] Mod 2.11: nn.Embedding for the target column "
                        "with a learnable unknown token for test rows.")
    p.add_argument("--n_perturbation_samples", type=int, default=1,
                   help="[mynanotabpfn] Perturbation draws averaged at inference "
                        "(default: 1). Set to 3 when --random_perturbations is active.")

    # mynanotabpfn — Priority 2 flags
    p.add_argument("--prenorm", action="store_true",
                   help="[mynanotabpfn] Mod 2.5: pre-norm LayerNorm in all sublayers "
                        "plus a final LayerNorm before the decoder.")
    p.add_argument("--cls_compression", action="store_true",
                   help="[mynanotabpfn] Mod 2.4: [CLS]-based row compression — "
                        "Stage 1 bi-attention + compress to CLS embeddings + "
                        "Stage 2 plain self-attention.")
    p.add_argument("--n_cls_tokens", type=int, default=2,
                   help="[mynanotabpfn] Mod 2.4: number of [CLS] tokens per row "
                        "(default: 2).")
    p.add_argument("--n_stage1_layers", type=int, default=3,
                   help="[mynanotabpfn] Mod 2.4: bi-attention layers before "
                        "compression (default: 3).")
    p.add_argument("--n_stage2_layers", type=int, default=3,
                   help="[mynanotabpfn] Mod 2.4: plain self-attention layers after "
                        "compression (default: 3).")
    p.add_argument("--icl_target_embedding", action="store_true",
                   help="[mynanotabpfn] Mod 2.4: inject class label into compressed "
                        "row embeddings before Stage 2 (TabICLv2 double injection).")

    # mynanotabpfn — Priority 3 flags
    p.add_argument("--feature_grouping", action="store_true",
                   help="[mynanotabpfn] Mod 2.2: circular grouped FeatureEncoder — "
                        "nn.Linear(3, d) instead of nn.Linear(1, d). Only meaningful "
                        "at m >= 7; degenerate at m=3 (training prior).")
    p.add_argument("--gated_residuals", action="store_true",
                   help="[mynanotabpfn] Mod 2.8: softplus-gated residual connections — "
                        "one scalar gate per sublayer, initialised to scale ≈1.")
    p.add_argument("--multi_layer_decoder", action="store_true",
                   help="[mynanotabpfn] Mod 2.9: concatenate intermediate-layer embeddings "
                        "before the decoder.")
    p.add_argument("--decoder_layer_indices", type=str, default=None,
                   help="[mynanotabpfn] Mod 2.9: comma-separated 1-based layer indices to "
                        "extract (e.g. '2,4,6'). Defaults to all layers when "
                        "--multi_layer_decoder is active.")

    # Mod 2.6 — feature subspace bagging (applies to nanotabpfn and mynanotabpfn)
    p.add_argument("--max_features", type=int, default=None,
                   help="Mod 2.6: max features per forward pass. When the dataset has "
                        "more features, random subsets of this size are averaged "
                        "(feature subspace bagging). Recommended value: 3 (training "
                        "distribution).")
    p.add_argument("--n_feature_subsets", type=int, default=None,
                   help="Mod 2.6: number of feature subsets to average. Defaults to "
                        "ceil(d / max_features) when --max_features is set.")

    # seldon-specific
    p.add_argument("--api_key", default=None,
                   help="[seldon] Neuralk API key (or set the NEURALK_API_KEY env var).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    return run_evaluation(model_factory=lambda: build_model(args), args=args)


if __name__ == "__main__":
    main()
