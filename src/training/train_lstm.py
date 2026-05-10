"""
train_lstm.py
=============
LSTM training entrypoint aligned with train_stgcn parameters:
- group-aware split (no leakage)
- optional label restriction/remapping (--labels)
- optional per-class caps (--cap_train/--cap_val)
- single train/val split (no KFold)
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report

from src.models.dataset import ProGaitDataset, pad_collate_fn
from src.models.model_lstm import GaitSequenceLSTM
import os

CLOUD_DATA_DIR = os.environ.get("VERTEX_DATA_DIR", None)

if CLOUD_DATA_DIR:
    MODELS_DIR = Path(CLOUD_DATA_DIR) / "models"
else:
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
    
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class RemappedSubset(torch.utils.data.Dataset):
    """Subset of an existing dataset with label remapping."""

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int], old_to_new: dict[int, int]):
        self.dataset = dataset
        self.indices = indices
        self.old_to_new = old_to_new

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        keypoints, metrics, ccc, issue = self.dataset[self.indices[idx]]
        old = int(issue.item())
        if old not in self.old_to_new:
            raise KeyError("Label id not in remap (sample should have been filtered out).")
        return keypoints, metrics, ccc, torch.tensor(self.old_to_new[old], dtype=torch.long)


def _extract_group_id(folder_name: str) -> str:
    return folder_name.rsplit("_clip_", 1)[0]


def split_indices_by_group(dataset: ProGaitDataset, val_split: float, seed: int):
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(dataset.valid_sessions):
        gid = _extract_group_id(s["folder"].name)
        groups.setdefault(gid, []).append(i)

    g_list = sorted(groups.keys())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(g_list), generator=generator).tolist()
    g_list = [g_list[i] for i in order]

    target_val = max(1, int(len(dataset) * val_split))

    val_idx: list[int] = []
    train_idx: list[int] = []
    for gid in g_list:
        idxs = groups[gid]
        if len(val_idx) < target_val:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    if not train_idx:
        moved = val_idx[-len(groups[g_list[-1]]):]
        val_idx = val_idx[:-len(moved)]
        train_idx = moved

    return train_idx, val_idx, groups


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return torch.clamp(weights, max=10.0)


def _subsample_indices_by_label(
    dataset: ProGaitDataset,
    indices: list[int],
    caps_by_class: dict[int, int],
    seed: int,
) -> list[int]:
    if not caps_by_class:
        return indices
    g = torch.Generator().manual_seed(seed)
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    perm = idx_tensor[torch.randperm(len(idx_tensor), generator=g)].tolist()
    selected: list[int] = []
    kept_counts: dict[int, int] = {k: 0 for k in caps_by_class}
    for i in perm:
        cls = dataset.class_mapping[dataset.valid_sessions[i]["issue_text"]]
        cap = caps_by_class.get(cls)
        if cap is None:
            selected.append(i)
            continue
        if kept_counts[cls] < cap:
            selected.append(i)
            kept_counts[cls] += 1
    return sorted(selected)


def _parse_cap_arg(text: str | None, dataset: ProGaitDataset) -> dict[int, int]:
    if not text:
        return {}
    caps: dict[int, int] = {}
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --cap format segment '{p}'. Use 'Label=NUM'.")
        label, num = [x.strip() for x in p.split("=", 1)]
        if label not in dataset.class_mapping:
            raise ValueError(
                f"Unknown label '{label}' in --cap. Valid labels: {list(dataset.class_mapping.keys())}"
            )
        caps[dataset.class_mapping[label]] = int(num)
    return caps


def _parse_labels_arg(text: str | None, dataset: ProGaitDataset) -> list[str]:
    if not text:
        return []
    labels = [p.strip() for p in text.split(",") if p.strip()]
    for lab in labels:
        if lab not in dataset.class_mapping:
            raise ValueError(
                f"Unknown label '{lab}' in --labels. Valid labels: {list(dataset.class_mapping.keys())}"
            )
    seen = set()
    out: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def _filter_indices_to_allowed_labels(
    dataset: ProGaitDataset,
    indices: list[int],
    allowed_old_label_ids: set[int],
) -> list[int]:
    if not allowed_old_label_ids:
        return indices
    filtered: list[int] = []
    for i in indices:
        cls = dataset.class_mapping[dataset.valid_sessions[i]["issue_text"]]
        if cls in allowed_old_label_ids:
            filtered.append(i)
    return filtered


class MetricFusedClassifier(nn.Module):
    def __init__(self, lstm_dim=64, metric_dim=6, head_dim=64, num_classes=5, dropout=0.3):
        super().__init__()
        self.metric_norm = nn.LayerNorm(metric_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim + metric_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, lstm_feats, raw_metrics):
        norm_metrics = self.metric_norm(raw_metrics)
        combined = torch.cat([lstm_feats, norm_metrics], dim=1)
        return self.fc(combined)


def train_one_epoch(
    encoder,
    classifier,
    loader,
    optimizer,
    criterion,
    device,
    max_batches=None,
):
    encoder.train()
    classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        keypoints, metrics, _, issues, lengths = batch
        keypoints = keypoints.to(device)
        metrics = metrics.to(device)
        issues = issues.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        lstm_out = encoder(keypoints, lengths)
        logits = classifier(lstm_out, metrics)

        loss = criterion(logits, issues)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(classifier.parameters()), 1.0
        )

        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == issues).sum().item()
        total += issues.numel()

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(1, len(loader))
    accuracy = (correct / total) * 100 if total else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    encoder,
    classifier,
    loader,
    criterion,
    device,
    max_batches=None,
):
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        keypoints, metrics, _, issues, lengths = batch
        keypoints = keypoints.to(device)
        metrics = metrics.to(device)
        issues = issues.to(device)
        lengths = lengths.to(device)

        lstm_out = encoder(keypoints, lengths)
        logits = classifier(lstm_out, metrics)

        loss = criterion(logits, issues)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == issues).sum().item()
        total += issues.numel()

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(1, len(loader))
    accuracy = (correct / total) * 100 if total else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_predictions(
    encoder,
    classifier,
    loader,
    device,
    max_batches=None,
):
    encoder.eval()
    classifier.eval()
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(loader):
        keypoints, metrics, _, issues, lengths = batch
        keypoints = keypoints.to(device)
        metrics = metrics.to(device)
        issues = issues.to(device)
        lengths = lengths.to(device)

        lstm_out = encoder(keypoints, lengths)
        logits = classifier(lstm_out, metrics)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(issues.tolist())
        y_pred.extend(preds.tolist())

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return y_true, y_pred


def run_training(
    epochs,
    lr,
    batch_size,
    val_split,
    seed,
    max_batches,
    cap_train,
    cap_val,
    labels,
    head_dim,
    dropout,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ProGaitDataset()

    selected_labels = _parse_labels_arg(labels, dataset)
    if selected_labels:
        allowed_old_ids = {dataset.class_mapping[n] for n in selected_labels}
        old_to_new = {dataset.class_mapping[n]: j for j, n in enumerate(selected_labels)}
        new_to_name = {j: n for n, j in zip(selected_labels, range(len(selected_labels)))}
        num_classes = len(selected_labels)
        print(f"Using restricted label set ({num_classes} classes): {selected_labels}")
    else:
        allowed_old_ids = set()
        old_to_new = {v: v for v in dataset.class_mapping.values()}
        new_to_name = {v: k for k, v in dataset.class_mapping.items()}
        num_classes = len(dataset.class_mapping)

    train_idx, val_idx, groups = split_indices_by_group(
        dataset=dataset,
        val_split=val_split,
        seed=seed,
    )

    if selected_labels:
        before_t, before_v = len(train_idx), len(val_idx)
        train_idx = _filter_indices_to_allowed_labels(dataset, train_idx, allowed_old_ids)
        val_idx = _filter_indices_to_allowed_labels(dataset, val_idx, allowed_old_ids)
        print(f"Filtered by labels: train {before_t}->{len(train_idx)} | val {before_v}->{len(val_idx)}")

    caps_train = _parse_cap_arg(cap_train, dataset)
    caps_val = _parse_cap_arg(cap_val, dataset)
    if caps_train:
        before = len(train_idx)
        train_idx = _subsample_indices_by_label(dataset, train_idx, caps_train, seed)
        print(f"Capped train set: {before} -> {len(train_idx)} clips")
    if caps_val:
        before = len(val_idx)
        val_idx = _subsample_indices_by_label(dataset, val_idx, caps_val, seed + 1)
        print(f"Capped val set: {before} -> {len(val_idx)} clips")

    print(
        f"Group split: {len(groups)} source videos | "
        f"train clips={len(train_idx)} | val clips={len(val_idx)}"
    )

    if selected_labels:
        train_ds = RemappedSubset(dataset, train_idx, old_to_new)
        val_ds = RemappedSubset(dataset, val_idx, old_to_new)
    else:
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    train_labels = [
        old_to_new[
            dataset.class_mapping[
                dataset.valid_sessions[i]["issue_text"]
            ]
        ]
        for i in train_idx
    ]

    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    print("\nClass Weights:")
    print(class_weights)

    encoder = GaitSequenceLSTM().to(device)
    classifier = MetricFusedClassifier(
        lstm_dim=64,
        metric_dim=6,
        head_dim=head_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    save_path = MODELS_DIR / "lstm_only.pth"
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            encoder,
            classifier,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches,
        )

        val_loss, val_acc = validate(
            encoder,
            classifier,
            val_loader,
            criterion,
            device,
            max_batches,
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.1f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.1f}%"
        )

        y_true, y_pred = evaluate_predictions(
            encoder=encoder,
            classifier=classifier,
            loader=val_loader,
            device=device,
            max_batches=max_batches,
        )

        labels_list = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels_list)
        target_names = [new_to_name.get(i, str(i)) for i in labels_list]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels_list,
            target_names=target_names,
            zero_division=0,
            digits=3,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "val_confusion_matrix": cm,
                    "val_report": report,
                },
                save_path,
            )
            print(f"✅ Saved best model -> {save_path}")

            reports_dir = MODELS_DIR / "lstm_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"best_epoch_{epoch:03d}_report.txt"
            cm_path = reports_dir / f"best_epoch_{epoch:03d}_cm.csv"
            report_path.write_text(report, encoding="utf-8")
            import numpy as np
            np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

            print("Validation classification report:")
            print(report)

    print("\nTraining Complete")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    return {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--cap_train",
        type=str,
        default="",
        help="Comma-separated per-class caps for train split. Example: 'Knee Issue=300,Unknown / Other=300'.",
    )
    parser.add_argument(
        "--cap_val",
        type=str,
        default="",
        help="Comma-separated per-class caps for val split. Same format as --cap_train.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated class names to restrict training/eval to (and remap to 0..K-1). Example: 'Knee Issue,Unknown / Other'.",
    )
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        max_batches=args.max_batches or None,
        cap_train=args.cap_train,
        cap_val=args.cap_val,
        labels=args.labels,
        head_dim=args.head_dim,
        dropout=args.dropout,
    )