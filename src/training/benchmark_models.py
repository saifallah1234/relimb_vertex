"""
benchmark_models.py
===================
Runs ST-GCN and LSTM training back-to-back with the same CLI parameters,
then prints a simple comparison of best validation accuracy.
"""

import argparse
from datetime import datetime

from src.training.train_lstm import run_training as run_lstm
from src.training.train_stgcn import run_training as run_stgcn


def benchmark(
    epochs: int,
    lr: float,
    batch_size: int,
    val_split: float,
    seed: int,
    max_batches: int | None,
    cap_train: str,
    cap_val: str,
    labels: str,
    head_dim: int,
    dropout: float,
) -> None:
    print("\n" + "=" * 55)
    print("  ReLimb Benchmark (ST-GCN vs LSTM)")
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 55)

    shared = dict(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        max_batches=max_batches,
        cap_train=cap_train,
        cap_val=cap_val,
        labels=labels,
    )

    print("\n─── ST-GCN ───────────────────────────────────────")
    metrics_stgcn = run_stgcn(**shared)

    print("\n─── LSTM ─────────────────────────────────────────")
    metrics_lstm = run_lstm(**shared, head_dim=head_dim, dropout=dropout)

    stgcn_acc = (metrics_stgcn or {}).get("best_val_acc", 0.0)
    lstm_acc = (metrics_lstm or {}).get("best_val_acc", 0.0)
    winner = "ST-GCN" if stgcn_acc > lstm_acc else "LSTM"

    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS  (best val accuracy)")
    print("=" * 55)
    print(f"  ST-GCN : {stgcn_acc:.2f}%")
    print(f"  LSTM   : {lstm_acc:.2f}%")
    print(f"  Winner : {winner}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReLimb model benchmark")
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
        help="Comma-separated class names to restrict training/eval to (and remap to 0..K-1).",
    )
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()

    benchmark(
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