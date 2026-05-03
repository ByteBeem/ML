"""
single_machine.py — Single-GPU / CPU baseline for comparison
=============================================================
Trains ResNet-50 on CIFAR-100 WITHOUT DDP.
Run first, then run train.py with torchrun, then compare with compare_metrics.py.

Usage:
    python single_machine.py [--epochs 30] [--batch_size 128] [--no_cuda] [--debug]
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CIFAR100Kaggle, TRAIN_TRANSFORM, VAL_TRANSFORM
from model import build_model
from utils import AverageMeter, accuracy, MetricsTracker


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers",  type=int,   default=None,
                   help="DataLoader workers. Defaults to 0 on Windows, 4 on Linux.")
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--no_cuda",      action="store_true")
    p.add_argument("--debug",        action="store_true",
                   help="Print extra debug messages every batch.")
    return p.parse_args()


def _default_workers():
    """0 on Windows (fork issues), 4 on Linux/Mac."""
    return 0 if sys.platform.startswith("win") else 4


# ─────────────────────────────────────────────
# Dataloaders
# ─────────────────────────────────────────────
def build_dataloaders(args):
    print(f"  [DEBUG] Building datasets …")
    train_ds = CIFAR100Kaggle(train=True,  transform=TRAIN_TRANSFORM)
    val_ds   = CIFAR100Kaggle(train=False, transform=VAL_TRANSFORM)

    nw = args.num_workers if args.num_workers is not None else _default_workers()
    print(f"  [DEBUG] DataLoader num_workers={nw}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=nw, pin_memory=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=nw, pin_memory=(nw > 0),
    )
    print(f"  [DEBUG] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler,
                device, epoch, total_epochs, debug=False):
    model.train()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    bar = tqdm(
        loader,
        desc=f"  Epoch {epoch+1:>3d}/{total_epochs} [Train]",
        unit="batch", ncols=100, leave=False,
    )

    for i, (images, targets) in enumerate(bar):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out  = model(images)
            loss = criterion(out, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        a1, a5 = accuracy(out, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)

        bar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.1f}%")

        if debug and (i % 50 == 0):
            print(f"    [DEBUG] batch {i}/{len(loader)} "
                  f"loss={losses.avg:.4f} acc1={top1.avg:.2f}%")

    return losses.avg, top1.avg, top5.avg


# ─────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs, debug=False):
    model.eval()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    bar = tqdm(
        loader,
        desc=f"  Epoch {epoch+1:>3d}/{total_epochs} [Val  ]",
        unit="batch", ncols=100, leave=False,
    )

    for i, (images, targets) in enumerate(bar):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out  = model(images)
            loss = criterion(out, targets)
        a1, a5 = accuracy(out, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)
        bar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.1f}%")

        if debug and (i % 50 == 0):
            print(f"    [DEBUG] val batch {i}/{len(loader)} "
                  f"loss={losses.avg:.4f} acc1={top1.avg:.2f}%")

    return losses.avg, top1.avg, top5.avg


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(
        "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("  SINGLE-MACHINE BASELINE — ResNet-50 on CIFAR-100")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  Platform    : {sys.platform}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Debug       : {args.debug}")
    print("=" * 65)

    train_loader, val_loader = build_dataloaders(args)

    print(f"  [DEBUG] Building model …")
    model     = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    use_amp = device.type == "cuda"
    # FIX: use the non-deprecated API (torch.amp.GradScaler)
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    print(f"  [DEBUG] AMP enabled: {use_amp}")

    tracker     = MetricsTracker()
    best_acc    = 0.0
    total_start = time.perf_counter()

    epoch_bar = tqdm(
        range(args.epochs),
        desc="  Overall progress",
        unit="epoch", ncols=100, position=0,
    )

    for epoch in epoch_bar:
        ep_start = time.perf_counter()

        if args.debug:
            print(f"\n  [DEBUG] === Epoch {epoch+1}/{args.epochs} start ===")

        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, args.epochs, debug=args.debug,
        )
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device,
            epoch, args.epochs, debug=args.debug,
        )
        scheduler.step()

        ep_time  = time.perf_counter() - ep_start
        best_acc = max(best_acc, val_acc1)
        lr_now   = scheduler.get_last_lr()[0]

        tracker.update(epoch, {
            "train_loss": round(train_loss, 4),
            "train_acc1": round(train_acc1, 2),
            "train_acc5": round(train_acc5, 2),
            "val_loss"  : round(val_loss, 4),
            "val_acc1"  : round(val_acc1, 2),
            "val_acc5"  : round(val_acc5, 2),
            "lr"        : round(lr_now, 6),
            "epoch_time": round(ep_time, 2),
        })

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            val=f"{val_acc1:.1f}%",
            best=f"{best_acc:.1f}%",
            t=f"{ep_time:.0f}s",
        )

        tqdm.write(
            f"  Epoch {epoch+1:>3d}/{args.epochs} | "
            f"LR {lr_now:.5f} | "
            f"Train {train_loss:.4f}/{train_acc1:.1f}% | "
            f"Val {val_loss:.4f}/{val_acc1:.1f}% | "
            f"Best {best_acc:.1f}% | "
            f"{ep_time:.1f}s"
        )

    total_time = time.perf_counter() - total_start

    results = {
        "mode"          : "single_machine",
        "world_size"    : 1,
        "device"        : str(device),
        "epochs"        : args.epochs,
        "batch_size"    : args.batch_size,
        "learning_rate" : args.lr,
        "total_time_sec": round(total_time, 2),
        "avg_epoch_sec" : round(total_time / args.epochs, 2),
        "best_val_acc1" : round(best_acc, 2),
        "history"       : tracker.history,
    }

    out = os.path.join(args.results_dir, "single_machine_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("  COMPLETE")
    print("=" * 65)
    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Avg/epoch   : {total_time/args.epochs:.2f}s")
    print(f"  Best Val@1  : {best_acc:.2f}%")
    print(f"  Results     : {out}")


if __name__ == "__main__":
    main()