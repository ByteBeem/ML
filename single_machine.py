"""
single_machine.py — Single-machine baseline (lightweight MLP on CIFAR-100)
==========================================================================
Usage:
    python single_machine.py [--epochs 10] [--batch_size 64] [--debug]
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=None)
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


def _default_workers():
    return 0 if sys.platform.startswith("win") else 2


def build_dataloaders(args):
    print(f"  [DEBUG] Loading datasets …")
    train_ds = CIFAR100Kaggle(train=True,  transform=TRAIN_TRANSFORM)
    val_ds   = CIFAR100Kaggle(train=False, transform=VAL_TRANSFORM)

    nw = args.num_workers if args.num_workers is not None else _default_workers()
    print(f"  [DEBUG] num_workers={nw}  batch_size={args.batch_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=nw)
    print(f"  [DEBUG] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, epoch, total, debug):
    model.train()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3d}/{total} [Train]",
               unit="batch", ncols=100, leave=False)

    for i, (images, targets) in enumerate(bar):
        images, targets = images.to(device), targets.to(device)

        out  = model(images)
        loss = criterion(out, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        a1, a5 = accuracy(out, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)

        bar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.1f}%")

        if debug and i % 100 == 0:
            print(f"    [DEBUG] batch {i}/{len(loader)} "
                  f"loss={losses.avg:.4f} acc1={top1.avg:.1f}%")

    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total, debug):
    model.eval()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3d}/{total} [Val  ]",
               unit="batch", ncols=100, leave=False)

    for i, (images, targets) in enumerate(bar):
        images, targets = images.to(device), targets.to(device)
        out  = model(images)
        loss = criterion(out, targets)
        a1, a5 = accuracy(out, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)
        bar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.1f}%")

    return losses.avg, top1.avg, top5.avg


def main():
    args   = parse_args()
    device = torch.device("cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 65)
    print("  SINGLE-MACHINE BASELINE — Lightweight MLP on CIFAR-100")
    print("=" * 65)
    print(f"  Device     : {device}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Debug      : {args.debug}")
    print("=" * 65)

    train_loader, val_loader = build_dataloaders(args)

    model     = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    tracker     = MetricsTracker()
    best_acc    = 0.0
    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        ep_start = time.perf_counter()
        if args.debug:
            print(f"\n  [DEBUG] === Epoch {epoch+1}/{args.epochs} ===")

        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.epochs, args.debug)
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion,
            device, epoch, args.epochs, args.debug)
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

        print(f"  Epoch {epoch+1:>3d}/{args.epochs} | "
              f"LR {lr_now:.5f} | "
              f"Train {train_loss:.4f}/{train_acc1:.1f}% | "
              f"Val {val_loss:.4f}/{val_acc1:.1f}% | "
              f"Best {best_acc:.1f}% | {ep_time:.1f}s")

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
    print(f"  Total time : {total_time:.2f}s  |  Avg/epoch : {total_time/args.epochs:.2f}s")
    print(f"  Best Val@1 : {best_acc:.2f}%  |  Results: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()