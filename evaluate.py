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

    # mynanotabpfn-specific (Priority 1 modification flags)
    p.add_argument("--target_aware", action="store_true",
                   help="[mynanotabpfn] Mod 2.1: add class embedding to feature cells "
                        "of training rows before the transformer layers.")
    p.add_argument("--random_perturbations", action="store_true",
                   help="[mynanotabpfn] Mod 2.3: add per-column random perturbations "
                        "to break feature-column symmetry.")
    p.add_argument("--target_encoder_use_embedding", action="store_true",
                   help="[mynanotabpfn] Mod 2.11: use nn.Embedding for the target "
                        "column instead of nn.Linear.")
    p.add_argument("--n_perturbation_samples", type=int, default=1,
                   help="[mynanotabpfn] Number of random-perturbation draws to average "
                        "at inference (default: 1). Set to 3 when --random_perturbations "
                        "is active.")

    # seldon-specific
    p.add_argument("--api_key", default=None,
                   help="[seldon] Neuralk API key (or set the NEURALK_API_KEY env var).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(model_factory=lambda: build_model(args), args=args)


if __name__ == "__main__":
    main()
