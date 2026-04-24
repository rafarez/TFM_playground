from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openml
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.preprocessing import LabelEncoder


BENCHMARKS: dict[str, str] = {
    # CLI name -> OpenML study alias
    "openml-cc18": "OpenML-CC18",
    # Additional benchmarks (tabarena, custom splits, ...) will be registered here.
}


@dataclass
class Task:
    id: int
    name: str
    X: np.ndarray
    y: np.ndarray
    classes: np.ndarray


def load_benchmark_tasks(name: str, cache_dir: str | None = None) -> list[int]:
    """Returns the OpenML task IDs of the named benchmark suite."""
    if name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}"
        )
    if cache_dir is not None:
        set_root_cache_directory(cache_dir)
    suite = openml.study.get_suite(BENCHMARKS[name])
    return list(suite.tasks)


def load_task(task_id: int) -> Task | None:
    """Loads an OpenML classification task and label-encodes its target.
    Returns `None` when the task is not a supervised classification task."""
    task = openml.tasks.get_task(task_id, download_splits=False)
    if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
        return None
    dataset = task.get_dataset(download_data=False)
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    le = LabelEncoder()
    y_enc = le.fit_transform(y.to_numpy())
    return Task(
        id=task_id,
        name=str(dataset.name),
        X=X.to_numpy(),
        y=y_enc,
        classes=le.classes_,
    )
