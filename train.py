"""
train.py — DDP distributed training (lightweight MLP on CIFAR-100)
===================================================================
Designed for low-RAM CPU-only machines (works on 1GB RAM per node).

Usage:
    # Machine 0 (master):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
      --master_addr=172.31.33.27 --master_port=29500 \
      train.py --debug

    # Machine 1:
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
      --master_addr=172.31.33.27 --master_port=29500 \
      train.py --debug
"""

import argparse
import json
import os
import sys
import socket
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import CIFAR100Kaggle, TRAIN_TRANSFORM, VAL_TRANSFORM
from model import build_model
from utils import (
    AverageMeter, accuracy, setup_logging,
    save_checkpoint, MetricsTracker,
)

import os
os.environ["USE_LIBUV"] = "0"


def parse_args():
    p = argparse.ArgumentParser(description="DDP Lightweight MLP — CIFAR-100")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="Per-rank batch size. Effective = batch_size × world_size")
    p.add_argument("--lr",           type=float, default=0.01,
                   help="Base LR — scaled by world_size automatically")
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=None)
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--resume",       default=None)
    p.add_argument("--save_every",   type=int,   default=5)
    p.add_argument("--log_interval", type=int,   default=100)
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


def _default_workers():
    return 0 if sys.platform.startswith("win") else 2


def build_dataloaders(args, rank, world_size):
    nw = args.num_workers if args.num_workers is not None else _default_workers()
    if rank == 0:
        print(f"  [DEBUG] Loading datasets  num_workers={nw} …")

    train_dataset = CIFAR100Kaggle(train=True,  transform=TRAIN_TRANSFORM)
    val_dataset   = CIFAR100Kaggle(train=False, transform=VAL_TRANSFORM)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=nw)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              sampler=val_sampler,   num_workers=nw)

    if rank == 0:
        print(f"  [DEBUG] Train batches/rank: {len(train_loader)}  "
              f"Val batches/rank: {len(val_loader)}")

    return train_loader, val_loader, train_sampler


def train_epoch(model, loader, criterion, optimizer, device,
                args, epoch, rank, logger):
    model.train()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    iterable = (tqdm(loader, desc=f"Epoch {epoch}", total=len(loader))
                if rank == 0 else loader)

    for i, (images, targets) in enumerate(iterable):
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

        if rank == 0:
            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{losses.avg:.4f}",
                                     acc1=f"{top1.avg:.1f}%")
            if args.debug or (i + 1) % args.log_interval == 0:
                msg = (f"  [DEBUG][Train] Epoch {epoch} "
                       f"Batch {i+1}/{len(loader)} | "
                       f"Loss {losses.avg:.4f} | Acc@1 {top1.avg:.1f}%")
                print(msg)
                logger.debug(msg)

    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(model, loader, criterion, device, rank, logger, debug=False):
    model.eval()
    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        out  = model(images)
        loss = criterion(out, targets)
        a1, a5 = accuracy(out, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1, bs)
        top5.update(a5, bs)

        if debug and rank == 0 and i % 50 == 0:
            print(f"    [DEBUG][Val] Batch {i}/{len(loader)} "
                  f"loss={losses.avg:.4f} acc1={top1.avg:.1f}%")

    # Average metrics across all ranks
    for meter in [losses, top1, top5]:
        t = torch.tensor(meter.avg, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        meter.avg = t.item()

    return losses.avg, top1.avg, top5.avg


def main():
    args = parse_args()

    # ── DDP init ────────────────────────────────────────────────────────────
    dist.init_process_group(backend="gloo", init_method="env://")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device("cpu")

    # ── Dirs ────────────────────────────────────────────────────────────────
    if rank == 0:
        for d in [args.results_dir, args.ckpt_dir]:
            os.makedirs(d, exist_ok=True)
            print(f"  [DEBUG] Directory ready: {d}")
    dist.barrier()

    # ── Logging ─────────────────────────────────────────────────────────────
    logger = setup_logging(rank, base_dir=".", debug=args.debug)

    if rank == 0:
        scaled_lr = args.lr * world_size
        print("=" * 70)
        print("  DISTRIBUTED MLP — PyTorch DDP + torchrun")
        print("=" * 70)
        print(f"  Platform         : {sys.platform}")
        print(f"  PyTorch          : {torch.__version__}")
        print(f"  World size       : {world_size} machine(s)")
        print(f"  Backend          : gloo (CPU-compatible)")
        print(f"  Host             : {socket.gethostname()}")
        print(f"  Epochs           : {args.epochs}")
        print(f"  Batch/rank       : {args.batch_size}  "
              f"(effective total: {args.batch_size * world_size})")
        print(f"  Base LR          : {args.lr}  →  scaled: {scaled_lr:.5f}")
        print(f"  Debug            : {args.debug}")
        print("=" * 70)

    try:
        # ── Data ────────────────────────────────────────────────────────────
        train_loader, val_loader, train_sampler = build_dataloaders(
            args, rank, world_size)

        # ── Model ───────────────────────────────────────────────────────────
        if rank == 0:
            print(f"  [DEBUG] Building model …")
        model = build_model(args.num_classes).to(device)
        model = DDP(model, find_unused_parameters=False)
        if rank == 0:
            print(f"  [DEBUG] Model wrapped in DDP")

        # ── Optimiser / Scheduler ───────────────────────────────────────────
        scaled_lr = args.lr * world_size
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=scaled_lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        # ── Resume ──────────────────────────────────────────────────────────
        start_epoch = 0
        best_acc    = 0.0
        if args.resume:
            if rank == 0:
                print(f"  [DEBUG] Resuming from {args.resume} …")
            ckpt = torch.load(args.resume, map_location=device)
            model.module.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc    = ckpt.get("best_acc", 0.0)
            if rank == 0:
                print(f"  [DEBUG] Resumed at epoch {start_epoch}, "
                      f"best_acc={best_acc:.2f}%")

        # ── Training loop ───────────────────────────────────────────────────
        tracker           = MetricsTracker()
        total_train_start = time.perf_counter()

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.perf_counter()
            train_sampler.set_epoch(epoch)

            if rank == 0 and args.debug:
                print(f"\n  [DEBUG] === Epoch {epoch}/{args.epochs-1} ===")

            train_loss, train_acc1, train_acc5 = train_epoch(
                model, train_loader, criterion, optimizer,
                device, args, epoch, rank, logger)

            val_loss, val_acc1, val_acc5 = validate(
                model, val_loader, criterion, device, rank, logger,
                debug=args.debug)

            scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            is_best    = val_acc1 > best_acc
            best_acc   = max(val_acc1, best_acc)

            tracker.update(epoch, {
                "train_loss": round(train_loss, 4),
                "train_acc1": round(train_acc1, 2),
                "train_acc5": round(train_acc5, 2),
                "val_loss"  : round(val_loss, 4),
                "val_acc1"  : round(val_acc1, 2),
                "val_acc5"  : round(val_acc5, 2),
                "lr"        : round(current_lr, 6),
                "epoch_time": round(epoch_time, 2),
            })

            if rank == 0:
                line = (
                    f"\n  Epoch {epoch:>3d}/{args.epochs-1} | "
                    f"LR {current_lr:.5f} | "
                    f"Train {train_loss:.4f}/{train_acc1:.1f}% | "
                    f"Val {val_loss:.4f}/{val_acc1:.1f}% | "
                    f"Best {best_acc:.1f}% | {epoch_time:.1f}s"
                    + (" ← best" if is_best else "")
                )
                print(line)
                logger.info(line.strip())

                if (epoch + 1) % args.save_every == 0 or is_best:
                    save_checkpoint(
                        state={
                            "epoch"    : epoch,
                            "model"    : model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_acc" : best_acc,
                            "args"     : vars(args),
                        },
                        is_best=is_best,
                        ckpt_dir=args.ckpt_dir,
                        epoch=epoch,
                    )

        # ── Save results ────────────────────────────────────────────────────
        total_time = time.perf_counter() - total_train_start

        if rank == 0:
            final = {
                "mode"           : "distributed_ddp",
                "world_size"     : world_size,
                "backend"        : "gloo",
                "device"         : str(device),
                "hostname"       : socket.gethostname(),
                "epochs"         : args.epochs,
                "batch_per_rank" : args.batch_size,
                "effective_batch": args.batch_size * world_size,
                "learning_rate"  : scaled_lr,
                "total_time_sec" : round(total_time, 2),
                "avg_epoch_sec"  : round(total_time / args.epochs, 2),
                "best_val_acc1"  : round(best_acc, 2),
                "history"        : tracker.history,
            }
            out = os.path.join(args.results_dir, "distributed_results.json")
            with open(out, "w") as f:
                json.dump(final, f, indent=2)

            print("\n" + "=" * 70)
            print("  TRAINING COMPLETE")
            print("=" * 70)
            print(f"  Total time : {total_time:.2f}s  |  "
                  f"Avg/epoch : {total_time/args.epochs:.2f}s")
            print(f"  Best Val@1 : {best_acc:.2f}%  |  Results: {out}")
            print("=" * 70)

    except Exception as exc:
        print(f"  [ERROR] Rank {rank} crashed: {exc}", flush=True)
        logger.exception(f"Rank {rank} fatal error")
        dist.destroy_process_group()
        raise

    dist.destroy_process_group()
    if rank == 0:
        print("  [DEBUG] Process group destroyed cleanly.")


if __name__ == "__main__":
    main()