from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tfmplayground.benchmark.benchmarks import load_benchmark_tasks, load_task
from tfmplayground.benchmark.metrics import compute_metrics
from tfmplayground.benchmark.models import BaseEvaluatedModel
from tfmplayground.benchmark.protocols import iter_splits


_METRIC_COLS = ("roc_auc", "accuracy", "log_loss")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _sanitize_args(args: argparse.Namespace) -> dict:
    cfg = dict(vars(args))
    if cfg.get("api_key"):
        cfg["api_key"] = "***redacted***"
    return cfg


def _resolve_task_ids(args: argparse.Namespace) -> list[int]:
    if args.task_ids:
        return [int(t) for t in args.task_ids.split(",")]
    return load_benchmark_tasks(args.benchmark, cache_dir=args.cache_dir)


def run_evaluation(
    *,
    model_factory: Callable[[], BaseEvaluatedModel],
    args: argparse.Namespace,
) -> Path:
    """Runs the evaluation described by `args` and writes all artifacts to a
    fresh timestamped subdirectory under `args.output_dir`. Returns its path."""
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = out_root / f"{stamp}_{args.model}"
    exp_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "args": _sanitize_args(args),
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    task_ids = _resolve_task_ids(args)
    print(f"Evaluating {args.model} on {len(task_ids)} tasks "
          f"({args.benchmark}, protocol={args.protocol}, seed={args.seed})")

    rows: list[dict] = []
    model = model_factory()
    for task_id in task_ids:
        try:
            task = load_task(
                task_id,
                max_n_features=args.max_n_features,
            )
        except Exception as e:
            print(f"[skip] task {task_id}: failed to load ({type(e).__name__}: {e})")
            continue
        if task is None:
            print(f"[skip] task {task_id}: not classification or exceeds size limits")
            continue

        classes = np.arange(len(task.classes))
        for fold_idx, (tr_idx, te_idx) in enumerate(
            iter_splits(task.X, task.y, protocol=args.protocol, seed=args.seed,
                        test_size=args.test_size)
        ):
            if args.max_n_samples is not None and len(tr_idx) > args.max_n_samples:
                tr_idx = tr_idx[:args.max_n_samples]

            X_tr, X_te = task.X[tr_idx], task.X[te_idx]
            y_tr, y_te = task.y[tr_idx], task.y[te_idx]
            base = {
                "task_id": task_id,
                "task_name": task.name,
                "fold": fold_idx,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_features": task.X.shape[1],
                "n_classes": len(task.classes),
            }

            t0 = time.perf_counter()
            try:
                model.fit(X_tr, y_tr)
                proba_chunks = [
                    model.predict_proba(X_te[start:start + args.batch_size])
                    for start in range(0, len(X_te), args.batch_size)
                ]
                y_proba = np.concatenate(proba_chunks, axis=0)
                y_pred = y_proba.argmax(axis=1)
                metrics = compute_metrics(
                    y_te, y_pred, y_proba, classes=classes, auc_only=args.auc_only
                )
                runtime = time.perf_counter() - t0
                row = {**base, "runtime_s": runtime, "error": None, **metrics}
                metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[ok]   {task.name} fold {fold_idx} "
                      f"({runtime:.1f}s) | {metric_str}")
            except Exception as e:
                runtime = time.perf_counter() - t0
                row = {**base, "runtime_s": runtime,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"[fail] {task.name} fold {fold_idx}: {row['error']}")
            rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(exp_dir / "raw.csv", index=False)

    metric_cols = [c for c in _METRIC_COLS if c in raw.columns]
    if metric_cols and len(raw):
        ok = raw[raw["error"].isna()]
        if len(ok):
            per_task = (
                ok.groupby(["task_id", "task_name"])[metric_cols]
                .agg(["mean", "std"])
            )
            per_task.columns = [f"{m}_{s}" for m, s in per_task.columns]
            per_task.reset_index().to_csv(exp_dir / "summary_per_task.csv", index=False)

            overall = ok[metric_cols].agg(["mean", "std"]).T
            overall.columns = ["mean", "std"]
            overall.to_csv(exp_dir / "summary_overall.csv")
            print("\n=== Overall (mean over all folds) ===")
            print(overall.to_string())

    print(f"\nResults written to {exp_dir}")
    return exp_dir
