"""Shared training loop for lightweight CNN and ResNet18 baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.dataset import IDX_TO_LABEL, LABEL_TO_IDX, create_dataloader
from src.metrics import find_best_threshold, predict_with_threshold
from src.models import MODEL_BUILDERS, count_parameters
from src.paths import PROJECT_ROOT


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = len(loader.dataset)
    return running_loss / n, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        mal_probs = probs[:, 1].cpu().numpy()
        preds = predict_with_threshold(mal_probs, threshold)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(mal_probs.tolist())

    n = len(loader.dataset)
    metrics = {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=[IDX_TO_LABEL[i] for i in range(2)],
            output_dict=True,
            zero_division=0,
        ),
        "labels": all_labels,
        "probs_malignant": all_probs,
    }
    if len(set(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    from sklearn.metrics import balanced_accuracy_score

    metrics["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)
    return metrics


def train_model(
    model_name: str,
    config: dict,
    output_path: str | Path,
) -> dict:
    cfg = config
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    device = torch.device(train_cfg.get("device", "cpu"))
    processed = PROJECT_ROOT / data_cfg["processed_dir"]

    train_loader = create_dataloader(
        processed / "train.csv",
        train_cfg["image_size"],
        train_cfg["batch_size"],
        shuffle=True,
        train=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = create_dataloader(
        processed / "val.csv",
        train_cfg["image_size"],
        train_cfg["batch_size"],
        shuffle=False,
        train=False,
        num_workers=train_cfg["num_workers"],
    )
    test_loader = create_dataloader(
        processed / "test.csv",
        train_cfg["image_size"],
        train_cfg["batch_size"],
        shuffle=False,
        train=False,
        num_workers=train_cfg["num_workers"],
    )

    train_df = pd.read_csv(processed / "train.csv")
    y_train = [LABEL_TO_IDX[l] for l in train_df["label"]]
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = MODEL_BUILDERS[model_name](num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
    )

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Model: {model_name} | params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Class weights (benign, malignant): {class_weights.round(3)}")

    best_val_ba = 0.0
    best_threshold = 0.5
    patience_counter = 0
    history = []
    start = time.time()

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        val_ba = val_metrics["balanced_accuracy"]
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_ba={val_ba:.3f} acc={val_metrics['accuracy']:.3f} f1={val_metrics['f1']:.3f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, **val_metrics})

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            patience_counter = 0
            best_threshold, _ = find_best_threshold(
                val_metrics["labels"],
                val_metrics["probs_malignant"],
            )
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "image_size": train_cfg["image_size"],
                    "label_map": IDX_TO_LABEL,
                    "val_balanced_accuracy": val_ba,
                    "threshold": best_threshold,
                },
                output_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stop_patience"]:
                print("Early stopping.")
                break

    elapsed = time.time() - start
    ckpt = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = ckpt.get("threshold", 0.5)
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=threshold)

    summary = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_time_sec": round(elapsed, 1),
        "best_val_balanced_accuracy": best_val_ba,
        "decision_threshold": threshold,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("labels", "probs_malignant")},
        "history": [
            {k: v for k, v in h.items() if k not in ("labels", "probs_malignant")} for h in history
        ],
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTest @ threshold={threshold:.2f} — acc={test_metrics['accuracy']:.3f} "
          f"ba={test_metrics['balanced_accuracy']:.3f} f1={test_metrics['f1']:.3f}")
    if "auc" in test_metrics:
        print(f"       auc={test_metrics['auc']:.3f}")
    print(f"Confusion matrix [benign/malignant rows]: {test_metrics['confusion_matrix']}")
    print(f"Saved checkpoint → {output_path}")
    return summary
