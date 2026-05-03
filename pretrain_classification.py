import argparse

import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback, WandbLoggerCallback
from tfmplayground.evaluation import TABARENA_TASKS, TOY_TASKS_CLASSIFICATION, get_openml_predictions
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.models.my_models import MyNanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

parser = argparse.ArgumentParser()
parser.add_argument("--priordump", type=str, default="50x3_3_100k_classification.h5", help="path to the prior dump")
parser.add_argument("--heads", type=int, default=6, help="number of attention heads")
parser.add_argument("--embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("--hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("--layers", type=int, default=6, help="number of transformer layers")
parser.add_argument(
    "--batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)"
)
parser.add_argument(
    "--accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights"
)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)"
)
parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--loadcheckpoint", type=str, default=None, help="checkpoint from which to continue training")
parser.add_argument("--multigpu", action="store_true", help="enable multi-GPU training using data parallelism")
parser.add_argument(
    "--runname",
    type=str,
    default="nanotabpfn",
    help="name of the training run, will be used to store the training checkpoints and for WandB logging",
)

# Priority 1 architectural modifications
parser.add_argument("--target_aware", action="store_true",
                    help="Mod 2.1: add class embedding to feature cells of training rows.")
parser.add_argument("--random_perturbations", action="store_true",
                    help="Mod 2.3: add per-column random perturbations to break feature symmetry.")
parser.add_argument("--target_encoder_use_embedding", action="store_true",
                    help="Mod 2.11: use nn.Embedding for the target column with an unknown token for test rows.")

# Priority 2 architectural modifications
parser.add_argument("--prenorm", action="store_true",
                    help="Mod 2.5: pre-norm LayerNorm in all sublayers plus a final LayerNorm before the decoder.")
parser.add_argument("--cls_compression", action="store_true",
                    help="Mod 2.4: [CLS]-based row compression with Stage 1 bi-attention and Stage 2 plain self-attention.")
parser.add_argument("--n_cls_tokens", type=int, default=2,
                    help="Mod 2.4: number of [CLS] tokens per row (default: 2).")
parser.add_argument("--n_stage1_layers", type=int, default=3,
                    help="Mod 2.4: bi-attention layers before row compression (default: 3). "
                         "Used only when --cls_compression is set; --layers is ignored.")
parser.add_argument("--n_stage2_layers", type=int, default=3,
                    help="Mod 2.4: plain self-attention layers after row compression (default: 3). "
                         "Used only when --cls_compression is set; --layers is ignored.")
parser.add_argument("--icl_target_embedding", action="store_true",
                    help="Mod 2.4: inject class label into compressed row embeddings before Stage 2.")

# Priority 3 architectural modifications
parser.add_argument("--feature_grouping", action="store_true",
                    help="Mod 2.2: circular grouped FeatureEncoder — nn.Linear(3, d) instead of nn.Linear(1, d). "
                         "Only meaningful at m >= 7; degenerate at m=3 (training prior).")
parser.add_argument("--gated_residuals", action="store_true",
                    help="Mod 2.8: scale each residual branch by softplus(gate), one scalar gate per sublayer.")
parser.add_argument("--multi_layer_decoder", action="store_true",
                    help="Mod 2.9: concatenate embeddings from multiple intermediate layers before the decoder.")
parser.add_argument("--decoder_layer_indices", type=str, default=None,
                    help="Mod 2.9: comma-separated 1-based layer indices to extract (e.g. '2,4,6'). "
                         "Defaults to all layers when --multi_layer_decoder is active.")

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = PriorDumpDataLoader(
    filename=args.priordump,
    num_steps=args.steps,
    batch_size=args.batchsize,
    device=device,
    starting_index=args.steps * (ckpt["epoch"] if ckpt else 0),
)

criterion = nn.CrossEntropyLoss()

model = MyNanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=prior.max_num_classes,
    # P1
    target_aware=args.target_aware,
    random_perturbations=args.random_perturbations,
    target_encoder_use_embedding=args.target_encoder_use_embedding,
    # P2
    prenorm=args.prenorm,
    cls_compression=args.cls_compression,
    n_cls_tokens=args.n_cls_tokens,
    n_stage1_layers=args.n_stage1_layers,
    n_stage2_layers=args.n_stage2_layers,
    icl_target_embedding=args.icl_target_embedding,
    # P3
    feature_grouping=args.feature_grouping,
    gated_residuals=args.gated_residuals,
    multi_layer_decoder=args.multi_layer_decoder,
    decoder_layer_indices=(
        [int(x) for x in args.decoder_layer_indices.split(",")]
        if args.decoder_layer_indices else None
    ),
)

if ckpt:
    model.load_state_dict(ckpt["model"])


class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for _dataset_name, (y_true, _y_pred, y_proba) in predictions.items():
            scores.append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        avg_score = sum(scores) / len(scores)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}",
            flush=True,
        )


class ProductionEvaluationLoggerCallback(WandbLoggerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        super().__init__(project, name, config, log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, classification=True, tasks=TABARENA_TASKS)
        scores = []
        log_metrics = {"epoch": epoch, "epoch_time": epoch_time, "mean_loss": loss}
        for dataset_name, (y_true, _y_pred, y_proba) in predictions.items():
            score = roc_auc_score(y_true, y_proba, multi_class="ovr")
            scores.append(score)
            log_metrics[f"roc_auc/{dataset_name}"] = score
        avg_score = sum(scores) / len(scores)
        log_metrics["tabarena_avg_roc_auc"] = avg_score
        self.wandb.log(log_metrics)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}",
            flush=True,
        )


# callbacks = [ProductionEvaluationLoggerCallback('nanoTFM', args.runname)]
callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
    multi_gpu=args.multigpu,
    run_name=args.runname,
)
