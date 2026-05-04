# MyNanoTabPFN (fork from TFM-Playground)

The purpose of this repository is to build upon the fully open source playground for tabular foundation models provided by the authors of TabPFNv2.
We start from a much smaller and simpler implementation of the TabPFNv2 architecture (nanoTabPFN) as well as a training loop, multiple interfaces to load prior data and an evaluation pipeline. We are planning to rapidly extend the repository with more features, prior interfaces and architectures.
From this, we added a evaluation script for different benchmarks, several modifications inspired by readings of TabICLv2 and TabPFNv2, and a notebook to run the full pretrain/test pipeline.

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

We offer the same interface as TabPFN:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### NanoTabPFN Code

- `tfmplayground/models/nanotabpfn.py` contains the implementation of the architecture in less than 300 lines of code. 
- `tfmplayground/models/my_models.py` implements an extension of the NanoTabPFN model with all the possible modifications we implemented. An in-depth explanation of these modifications can be found below in this document.
- `tfmplayground/train.py` implements a simple training loop in under 200 lines and `tfmplayground/external_priors/` provides an interface to publicly available priors form other repositories as well as a dataloader for loading HDF5 dumps.
- `tfmplayground/benchmarks/` provides all the elements for in-depth evaluation of our models for several metrics, easy inclusion of new benchmarks, and an abstract model builder that adapts to models not produced by this codebase, such as Seldon from Neuralk.
- `pipeline_notebook.ipynb` gives a line-by-line rundown on how to pretrain and test a model described in the experiments file.

### Pretrain your own small nanoTabPFN
First we download 100k pre-generated datasets with 50 datapoints, 3 features and up to 3 classes each from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5).

Then you can run:
```
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 --priordump 50x3_3_100k_classification.h5
```
This should take less than 5 min on a modern NVIDIA GPU (around 10 minutes on Macbook M4 Pro GPU and around 40 min on M4 Pro CPU).


#### Step by Step Explanation (Classifier)

First we import our Architecture, Prior interface and training loop, etc.
```python
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss
```
then we instantiate our model and loss criterion:
```python
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=10,
)
criterion = CrossEntropyLoss()
```
then we instantiate our prior:
```python
device = get_default_device()
prior = PriorDumpDataLoader(filename='50x3_3_100k_classification.h5', num_steps=25, batch_size=50, device=device)
```
and finally train our model:
```python
trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=80,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)
```

---

## Run the pretrain/test pipeline on Google Colab

Follow the instructions of `pipeline_notebook.ipynb` to run the full pipeline on a Colab server.

## Architectural Modifications (MyNanoTabPFN)

`tfmplayground/models/my_models.py` contains `MyNanoTabPFNModel`, an extended version of nanoTabPFN that supports all modifications described below via opt-in flags. All flags default to `False`/`None`, reproducing the baseline exactly when no flags are set.

### Training with modifications

`pretrain_classification.py` now uses `MyNanoTabPFNModel` and accepts one flag per modification:

```bash
# Baseline (unchanged behaviour)
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 \
  --priordump 50x3_3_100k_classification.h5

# All Priority 1 modifications
python pretrain_classification.py --target_aware --random_perturbations \
  --target_encoder_use_embedding --priordump 50x3_3_100k_classification.h5

# Pre-norm + CLS row compression (6 total layers: 3 Stage-1 + 3 Stage-2)
python pretrain_classification.py --prenorm --cls_compression \
  --n_stage1_layers 3 --n_stage2_layers 3 --priordump 50x3_3_100k_classification.h5
```

### Encoder-based modifications

| Flag | Modification code | Description |
|---|---|---|
| `--target_aware` | TAE | Add a learnable class embedding to every feature cell of each training row before the transformer layers. Each class gets a free `d`-dim vector (separate from the TargetEncoder). Gives feature attention direct access to class-conditional patterns from layer 1. Source: TabICLv2. |
| `--random_perturbations` | RP | Add a per-column random perturbation to break feature-column symmetry. Re-sampled every forward pass. At inference, average over `--n_perturbation_samples` draws (default 1; set to 3 when active). Source: TabPFN v2 / Ye et al. 2025. |
| `--target_encoder_use_embedding` | TE | Replace the TargetEncoder's `nn.Linear(1, d)` with `nn.Embedding(C+1, d)`. Training rows look up their true class index; test rows use a dedicated learnable "unknown / predict-me" token at index `C`. Eliminates false ordinated structure on class indices. |

### Transfomer_based modifications

| Flag | Modification code | Description |
|---|---|---|
| `--prenorm` | PN | Switch all three sublayers (feature-attn, datapoint-attn, MLP) from post-norm to pre-norm. Adds a final `LayerNorm(d)` after the last transformer layer, before the decoder. Prerequisite for stable training at greater depth. Source: TabICLv2. |
| `--cls_compression` | CLS | Split the transformer into Stage 1 (bi-attention with K CLS tokens, `--n_stage1_layers`) and Stage 2 (plain self-attention on compressed row embeddings, `--n_stage2_layers`). After Stage 1 the K CLS outputs per row are concatenated into a `K·d`-dim row embedding; the decoder reads this. Use `--icl_target_embedding` to inject a second class label (separate embedding) into the compressed row embeddings before Stage 2. Source: TabICLv2. |


### Miscellaneous modifications

| Flag | Modification code | Description |
|---|---|---|
| `--max_features` | MaxF | Feature subspace bagging (Inference-time only.): when the dataset has more features than `--max_features`, randomly sample `ceil(d / max_features)` subsets of that size, run inference on each, and average the predicted probabilities. Recommended value: 3 (the training distribution). |
| `--feature_grouping` | FG | Replace the per-scalar `nn.Linear(1, d)` in FeatureEncoder with a circular grouped encoding: for column `j` the input is `[x_j, x_{(j+1)%m}, x_{(j+3)%m}]` and a shared `nn.Linear(3, d)` is applied. Degenerate at `m=3` (every group contains all 3 features); meaningful only at `m ≥ 7`. Source: TabICLv2. |
| `--gated_residuals` | GR | Replace `x + sublayer(x)` with `x + softplus(gate) · sublayer(x)` in every sublayer. One scalar `gate` per sublayer per layer, initialised so `softplus(gate) ≈ 1` (baseline scale at init). |
| `--multi_layer_decoder` | DMLP | Extract pre-decoder embeddings from multiple intermediate transformer layers and concatenate them before the decoder. Specify layers with `--decoder_layer_indices 2,4,6` (1-based); defaults to all layers. The decoder input dimension scales accordingly. Source: Ye et al. 2025. |

Leave-one-fold-out feature extraction is inference-time only and is exposed as `NanoTabPFNClassifier.extract_features(X, y, n_folds=10)` — it requires no training flag.

---

## Evaluating models

`evaluate.py` is the benchmark harness. It runs any supported model on a set of OpenML tasks and writes per-fold metrics (ROC AUC, accuracy, log-loss) together with a config snapshot to a timestamped experiment folder.

```bash
# Pretrained baseline on OpenML-CC18 (fixed 80/20 split, all three metrics)
python evaluate.py --model nanotabpfn --benchmark openml-cc18

# Pretrained baseline, 5-fold CV, AUC only
python evaluate.py --model nanotabpfn --benchmark openml-cc18 --protocol 5fold --auc_only

# Custom-trained MyNanoTabPFN checkpoint with TAE+TE+RP mods + perturbation averaging
python evaluate.py --model mynanotabpfn --checkpoint workdir/myrun/latest_checkpoint.pth \
  --target_aware --random_perturbations --target_encoder_use_embedding \
  --n_perturbation_samples 3


# Feature subspace bagging on the released baseline (m=3 training distribution)
python evaluate.py --model nanotabpfn --max_features 3

# Neuralk Seldon API
python evaluate.py --model seldon --api_key nk_live_xxx

# Restrict to specific OpenML task IDs
python evaluate.py --model nanotabpfn --task_ids 59,2382,9946
```

### Output structure

Each run creates `experiments/YYYYMMDD_HHMMSS_<model>/` containing:

| File | Contents |
|---|---|
| `config.json` | All CLI arguments (API key redacted) + current git commit SHA |
| `raw.csv` | One row per (task × fold): task metadata, metrics, runtime, error message if any |
| `summary_per_task.csv` | Mean ± std of each metric aggregated over folds, per task |
| `summary_overall.csv` | Mean ± std across all tasks and folds |

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `nanotabpfn` | `nanotabpfn`, `mynanotabpfn`, or `seldon` |
| `--benchmark` | `tabarena` | OpenML suite name; use `openml-cc18` for the CC18 benchmark |
| `--task_ids` | — | Comma-separated OpenML task IDs; overrides `--benchmark` |
| `--protocol` | `fixed` | `fixed` (stratified 80/20 split) or `5fold` |
| `--seed` | `42` | Random seed for splits |
| `--auc_only` | off | Only compute ROC AUC, skip accuracy and log-loss |
| `--batch_size` | `128` | Test-set chunk size for `predict_proba`; lower if CUDA OOM |
| `--num_mem_chunks` | `8` | Attention chunking factor (see bug fix below) |
| `--max_n_features` | `500` | Skip tasks with more features than this |
| `--max_n_samples` | `10000` | Skip tasks with more instances than this |
| `--max_features` | — | Mod 2.6: feature subspace bagging subset size |
| `--checkpoint` | — | Path to a `.pth` checkpoint; auto-detects baseline vs. modified architecture |

All modification flags from `pretrain_classification.py` (`--target_aware`, `--prenorm`, `--cls_compression`, etc.) are also accepted by `evaluate.py` when `--model mynanotabpfn` is used.

---

## Bug fix: `num_mem_chunks` not forwarded to transformer blocks (commit 150a61f)

**The bug.** `NanoTabPFNModel._forward` accepted a `num_mem_chunks` keyword argument and stored it, but called each transformer block as:

```python
src = block(src, train_test_split_index=train_test_split_index)
```

`TransformerEncoderLayer.forward` has signature `forward(src, train_test_split_index, num_mem_chunks=1)`. Because `num_mem_chunks` was never forwarded, the `memory_chunking` decorator inside every layer always ran with `num_mem_chunks=1` — the unchunked path. The `--num_mem_chunks` CLI flag was entirely ignored at inference.

**The fix** (one line, `tfmplayground/models/nanotabpfn.py`):

```python
# Before:
src = block(src, train_test_split_index=train_test_split_index)

# After:
src = block(src, train_test_split_index=train_test_split_index, num_mem_chunks=num_mem_chunks)
```

