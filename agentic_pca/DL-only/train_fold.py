#!/usr/bin/env python3
"""Train one outer fold and perform one-time inference on its held-out patients."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from common import (
    CLASS_NAMES,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    dependency_versions,
    load_json,
    sha256_file,
    sha256_json,
    utc_now,
    validate_manifest,
)
from data import PairedPETDataset, seed_worker
from models import SUPPORTED_MODELS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test exactly one fold of the dual-PET classifier."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_CACHE_DIR / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="swin_tiny_3d",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--class-weight-beta", type=float, default=0.99)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cuda', 'cuda:N', or 'cpu'.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "none"),
        default="bfloat16",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable Swin activation checkpointing.",
    )
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-temperature-scaling", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest, split, cache samples and model without training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an already completed result for this fold.",
    )
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def autocast_context(
    device: torch.device,
    amp_dtype: str,
) -> contextlib.AbstractContextManager:
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def effective_number_weights(
    labels: Sequence[int],
    num_classes: int,
    beta: float,
) -> torch.Tensor:
    if not 0 <= beta < 1:
        raise ValueError("--class-weight-beta must satisfy 0 <= beta < 1.")
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing a class: {counts.tolist()}")
    if beta == 0:
        weights = np.ones_like(counts)
    else:
        effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective
        weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def classification_metrics(
    true: Sequence[int],
    predicted: Sequence[int],
) -> dict[str, Any]:
    true_array = np.asarray(true, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_array,
        predicted_array,
        labels=labels,
        zero_division=0,
    )
    return {
        "patients": int(true_array.size),
        "accuracy": float(accuracy_score(true_array, predicted_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_array, predicted_array)
        ),
        "macro_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            true_array,
            predicted_array,
            labels=labels,
        ).tolist(),
    }


def make_loader(
    dataset: PairedPETDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: str,
    accumulation_steps: int,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        with autocast_context(device, amp_dtype):
            logits = model(images)
            raw_loss = criterion(logits, labels)
            loss = raw_loss / group_size
        scaler.scale(loss).backward()
        end_of_group = (step - group_start + 1) == group_size
        if end_of_group:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        total_loss += float(raw_loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.inference_mode()
def evaluate_labeled(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: str,
    use_tta: bool = False,
) -> dict[str, Any]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    case_ids: list[str] = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            logits = model(images)
            if use_tta:
                flipped = model(torch.flip(images, dims=(2,)))
                logits = (logits + flipped) / 2.0
            loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss) * batch_size
        total_samples += batch_size
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.cpu())
        case_ids.extend(str(case_id) for case_id in batch["case_id"])
    logits_array = torch.cat(logits_all).numpy()
    labels_array = torch.cat(labels_all).numpy()
    metrics = classification_metrics(labels_array, logits_array.argmax(axis=1))
    return {
        "loss": total_loss / total_samples,
        "logits": logits_array,
        "labels": labels_array,
        "case_ids": case_ids,
        "metrics": metrics,
    }


@torch.inference_mode()
def predict_unlabeled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    use_tta: bool,
) -> tuple[list[str], np.ndarray]:
    model.eval()
    case_ids: list[str] = []
    logits_all: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            logits = model(images)
            if use_tta:
                flipped = model(torch.flip(images, dims=(2,)))
                logits = (logits + flipped) / 2.0
        logits_all.append(logits.float().cpu())
        case_ids.extend(str(case_id) for case_id in batch["case_id"])
    return case_ids, torch.cat(logits_all).numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive temperature using only the inner validation split."""
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0))


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )


def main() -> int:
    args = parse_args()
    for name in (
        "epochs",
        "patience",
        "batch_size",
        "accumulation_steps",
    ):
        if getattr(args, name.replace("-", "_")) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs).")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError(
            "Learning rates must satisfy 0 < min-learning-rate <= learning-rate."
        )
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be positive.")

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    manifest_fingerprint = manifest.get("manifest_fingerprint")
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint:
        raise ValueError("Manifest is missing manifest_fingerprint.")
    preprocessing = manifest["preprocessing"]
    preprocess_fingerprint = preprocessing["fingerprint"]
    target_shape = tuple(int(value) for value in preprocessing["target_shape_xyz"])

    fold_definition = manifest["folds"][args.fold]
    records_by_id = {
        str(record["case_id"]): record for record in manifest["patients"]
    }

    def records_for(split: str) -> list[Mapping[str, Any]]:
        key = f"{split}_case_ids"
        return [records_by_id[case_id] for case_id in fold_definition[key]]

    train_records = records_for("train")
    val_records = records_for("val")
    test_records = records_for("test")
    train_ids = {record["case_id"] for record in train_records}
    val_ids = {record["case_id"] for record in val_records}
    test_ids = {record["case_id"] for record in test_records}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("The selected fold has split overlap.")
    if train_ids | val_ids | test_ids != set(records_by_id):
        raise ValueError("The selected fold does not cover the complete cohort.")

    fold_seed = args.seed + args.fold
    set_determinism(fold_seed)
    device = resolve_device(args.device)
    if device.type != "cuda" and args.amp_dtype != "none":
        print("AMP disabled because the selected device is not CUDA.", flush=True)
    model = build_model(
        args.model,
        num_classes=len(CLASS_NAMES),
        use_checkpoint=not args.no_checkpointing,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    fold_dir = args.output_dir.resolve() / "folds" / f"fold_{args.fold}"
    completion_path = fold_dir / "DONE.json"
    if completion_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{completion_path} already exists; use --overwrite to rerun this fold."
        )
    if completion_path.exists() and args.overwrite:
        completion_path.unlink()

    split_document = {
        "schema_version": SCHEMA_VERSION,
        "fold": args.fold,
        "manifest_fingerprint": manifest_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "train_case_ids": [record["case_id"] for record in train_records],
        "val_case_ids": [record["case_id"] for record in val_records],
        "test_case_ids": [record["case_id"] for record in test_records],
        "class_counts": fold_definition["class_counts"],
    }
    atomic_write_json(fold_dir / "split.json", split_document)

    training_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "fold": args.fold,
        "fold_seed": fold_seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_fingerprint": manifest_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "classes": list(CLASS_NAMES),
        "model": model.model_config,
        "pretrained": False,
        "training": {
            "base_seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "workers": args.workers,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "loss": "effective-number weighted cross entropy",
            "class_weight_beta": args.class_weight_beta,
            "label_smoothing": args.label_smoothing,
            "gradient_clip": args.gradient_clip,
            "amp_dtype": (
                args.amp_dtype if device.type == "cuda" else "none"
            ),
            "selection_metric": "validation macro-F1; validation loss tie-break",
            "test_time_augmentation": not args.no_tta,
            "temperature_scaling_fit_on_validation": (
                not args.no_temperature_scaling
            ),
        },
        "device": str(device),
        "dependencies": dependency_versions(),
        "parameter_count": parameter_count,
    }
    training_config["protocol_fingerprint"] = sha256_json(
        {
            "manifest_fingerprint": manifest_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
            "classes": list(CLASS_NAMES),
            "model": training_config["model"],
            "pretrained": training_config["pretrained"],
            "training": training_config["training"],
            "dependencies": training_config["dependencies"],
        }
    )
    training_config["training_fingerprint"] = sha256_json(
        {
            key: value
            for key, value in training_config.items()
            if key != "created_at_utc"
        }
    )
    atomic_write_json(fold_dir / "config.json", training_config)

    # A dry run still checks that representative cached tensors and the model
    # interface agree, but it does not allocate a full training graph.
    if args.dry_run:
        for split_name, records in (
            ("train", train_records),
            ("val", val_records),
            ("test", test_records),
        ):
            sample = PairedPETDataset(
                records[:1],
                target_shape=target_shape,
                include_label=split_name != "test",
            )[0]
            if tuple(sample["image"].shape) != (2, *target_shape):
                raise AssertionError("Unexpected cached sample shape.")
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "fold": args.fold,
                    "split_sizes": {
                        "train": len(train_records),
                        "val": len(val_records),
                        "test": len(test_records),
                    },
                    "model": model.model_config,
                    "parameter_count": parameter_count,
                    "device": str(device),
                },
                indent=2,
            )
        )
        return 0

    model.to(device)
    train_dataset = PairedPETDataset(
        train_records,
        target_shape=target_shape,
        augment=True,
        include_label=True,
    )
    val_dataset = PairedPETDataset(
        val_records,
        target_shape=target_shape,
        augment=False,
        include_label=True,
    )
    # The held-out Dataset deliberately returns no labels. Ground truth is
    # joined only after predictions have been serialized in a blinded file.
    test_dataset = PairedPETDataset(
        test_records,
        target_shape=target_shape,
        augment=False,
        include_label=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        seed=fold_seed,
        **loader_kwargs,
    )
    val_loader = make_loader(
        val_dataset,
        shuffle=False,
        seed=fold_seed + 10_000,
        **loader_kwargs,
    )
    test_loader = make_loader(
        test_dataset,
        shuffle=False,
        seed=fold_seed + 20_000,
        **loader_kwargs,
    )

    train_labels = [int(record["label_index"]) for record in train_records]
    class_weights = effective_number_weights(
        train_labels,
        len(CLASS_NAMES),
        args.class_weight_beta,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_multiplier(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs - 1
        )
        minimum_ratio = args.min_learning_rate / args.learning_rate
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.amp_dtype == "float16",
    )
    best_path = fold_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp_dtype,
            args.accumulation_steps,
            args.gradient_clip,
        )
        validation = evaluate_labeled(
            model,
            val_loader,
            criterion,
            device,
            args.amp_dtype,
        )
        macro_f1 = float(validation["metrics"]["macro_f1"])
        val_loss = float(validation["loss"])
        improved = macro_f1 > best_macro_f1 + 1e-12 or (
            math.isclose(macro_f1, best_macro_f1, abs_tol=1e-12)
            and val_loss < best_val_loss
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": validation["metrics"]["accuracy"],
            "val_balanced_accuracy": validation["metrics"]["balanced_accuracy"],
            "val_macro_f1": macro_f1,
            "improved": int(improved),
        }
        history.append(row)
        atomic_write_csv(fold_dir / "history.csv", list(row), history)

        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "epoch": epoch + 1,
            "fold": args.fold,
            "model_name": args.model,
            "model_config": model.model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "manifest_fingerprint": manifest_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
            "training_fingerprint": training_config["training_fingerprint"],
            "protocol_fingerprint": training_config["protocol_fingerprint"],
            "validation": {
                "loss": val_loss,
                "metrics": validation["metrics"],
            },
        }
        atomic_torch_save(fold_dir / "last_model.pt", checkpoint)
        if improved:
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            atomic_torch_save(best_path, checkpoint)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        print(
            f"fold={args.fold} epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_result['loss']:.5f} "
            f"val_loss={val_loss:.5f} val_macro_f1={macro_f1:.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epoch + 1} epochs "
                f"(patience={args.patience}).",
                flush=True,
            )
            break

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    if (
        best_checkpoint["manifest_fingerprint"] != manifest_fingerprint
        or best_checkpoint["preprocess_fingerprint"] != preprocess_fingerprint
        or best_checkpoint["training_fingerprint"]
        != training_config["training_fingerprint"]
        or best_checkpoint["protocol_fingerprint"]
        != training_config["protocol_fingerprint"]
    ):
        raise ValueError("Best checkpoint fingerprint mismatch.")
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    validation = evaluate_labeled(
        model,
        val_loader,
        criterion,
        device,
        args.amp_dtype,
        use_tta=not args.no_tta,
    )
    temperature = (
        1.0
        if args.no_temperature_scaling
        else fit_temperature(validation["logits"], validation["labels"])
    )

    test_case_ids, test_logits = predict_unlabeled(
        model,
        test_loader,
        device,
        args.amp_dtype,
        use_tta=not args.no_tta,
    )
    probabilities = torch.softmax(
        torch.tensor(test_logits / temperature),
        dim=1,
    ).numpy()
    predicted_indices = probabilities.argmax(axis=1)
    checkpoint_sha256 = sha256_file(best_path)

    blinded_rows: list[dict[str, Any]] = []
    for case_id, predicted_index, probability in zip(
        test_case_ids,
        predicted_indices,
        probabilities,
        strict=True,
    ):
        blinded_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "case_id": case_id,
                "fold": args.fold,
                "predicted_index": int(predicted_index),
                "predicted_class": CLASS_NAMES[int(predicted_index)],
                "probabilities": {
                    name: float(probability[index])
                    for index, name in enumerate(CLASS_NAMES)
                },
                "temperature": temperature,
                "checkpoint_sha256": checkpoint_sha256,
                "preprocess_fingerprint": preprocess_fingerprint,
                "training_fingerprint": training_config["training_fingerprint"],
                "protocol_fingerprint": training_config[
                    "protocol_fingerprint"
                ],
            }
        )
    blinded_path = fold_dir / "test_predictions_blinded.jsonl"
    atomic_write_text(blinded_path, rows_to_jsonl(blinded_rows))

    expected_test_ids = set(fold_definition["test_case_ids"])
    if set(test_case_ids) != expected_test_ids or len(test_case_ids) != len(
        expected_test_ids
    ):
        raise ValueError("Test inference IDs do not exactly match the frozen split.")
    prediction_rows: list[dict[str, Any]] = []
    for row in blinded_rows:
        source = records_by_id[row["case_id"]]
        truth = int(source["label_index"])
        prediction_rows.append(
            {
                **row,
                "true_index": truth,
                "true_label": source["label"],
                "correct": int(row["predicted_index"]) == truth,
            }
        )
    atomic_write_text(
        fold_dir / "test_predictions.jsonl",
        rows_to_jsonl(prediction_rows),
    )
    test_metrics = classification_metrics(
        [row["true_index"] for row in prediction_rows],
        [row["predicted_index"] for row in prediction_rows],
    )
    fold_metrics = {
        "schema_version": SCHEMA_VERSION,
        "fold": args.fold,
        "protocol_fingerprint": training_config["protocol_fingerprint"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_macro_f1": best_macro_f1,
        "temperature": temperature,
        "validation": {
            "loss": validation["loss"],
            "metrics": validation["metrics"],
        },
        "test_evaluated_after_model_selection": True,
        "test": test_metrics,
        "class_weights": {
            name: float(class_weights[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "checkpoint": str(best_path),
        "checkpoint_sha256": checkpoint_sha256,
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    atomic_write_json(fold_dir / "metrics.json", fold_metrics)
    atomic_write_json(
        completion_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "fold": args.fold,
            "test_patients": len(prediction_rows),
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_fingerprint": manifest_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
            "protocol_fingerprint": training_config["protocol_fingerprint"],
        },
    )
    print(
        f"Fold {args.fold} complete: test_accuracy={test_metrics['accuracy']:.4f}, "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
