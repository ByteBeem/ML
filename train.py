import argparse
import json
import os
import time
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import CIFAR100Kaggle, TRAIN_TRANSFORM, VAL_TRANSFORM
from model import build_model
from utils import (
    AverageMeter, accuracy, setup_logging,
    save_checkpoint, MetricsTracker
)


# ARGS
def parse_args():
    p = argparse.ArgumentParser(description="DDP Image Recognition")
    p.add_argument("--data_dir",    default="data",         help="Dataset root")
    p.add_argument("--results_dir", default="results",      help="Where to save JSON results")
    p.add_argument("--ckpt_dir",    default="checkpoints",  help="Checkpoint directory")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=128,  help="Per-GPU batch size")
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=0,    help="0 = safest on Windows")
    p.add_argument("--num_classes", type=int, default=100,  help="100 for CIFAR-100")
    p.add_argument("--no_cuda",     action="store_true",    help="Force CPU (testing only)")
    p.add_argument("--resume",      default=None,           help="Path to checkpoint to resume from")
    p.add_argument("--save_every",  type=int, default=5,    help="Save checkpoint every N epochs")
    p.add_argument("--log_interval",type=int, default=50,   help="Log every N batches")
    return p.parse_args()



# DATASET
def build_dataloaders(args, rank, world_size):
    # Only rank 0 prints the download message; others load from cache silently
    train_dataset = CIFAR100Kaggle(train=True,  transform=TRAIN_TRANSFORM)
    val_dataset   = CIFAR100Kaggle(train=False, transform=VAL_TRANSFORM)

    # DistributedSampler ensures each rank sees a unique non-overlapping
    # shard of the data → total effective batch = batch_size × world_size
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, val_loader, train_sampler



# TRAIN ONE EPOCH
def train_epoch(model, loader, criterion, optimizer, scaler, device, args, epoch, rank):
    model.train()

    losses   = AverageMeter("Loss")
    top1     = AverageMeter("Acc@1")
    top5     = AverageMeter("Acc@5")
    batch_time = AverageMeter("Time")

    end = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed precision forward
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(images)
            loss   = criterion(output, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()       # DDP all-reduces gradients here
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(acc1, bs)
        top5.update(acc5, bs)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if rank == 0 and (i + 1) % args.log_interval == 0:
            print(
                f"  [Train] Epoch {epoch:>3d} | "
                f"Batch {i+1:>4d}/{len(loader)} | "
                f"Loss {losses.avg:.4f} | "
                f"Acc@1 {top1.avg:.2f}% | "
                f"Acc@5 {top5.avg:.2f}% | "
                f"Batch {batch_time.avg*1000:.1f}ms"
            )

    return losses.avg, top1.avg, top5.avg


# VALIDATE

@torch.no_grad()
def validate(model, loader, criterion, device, rank):
    model.eval()

    losses = AverageMeter("Loss")
    top1   = AverageMeter("Acc@1")
    top5   = AverageMeter("Acc@5")

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(images)
            loss   = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(acc1, bs)
        top5.update(acc5, bs)

    # Aggregate metrics across all ranks
    for meter in [losses, top1, top5]:
        t = torch.tensor(meter.avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        meter.avg = t.item()

    return losses.avg, top1.avg, top5.avg


# MAIN

def main():
    args = parse_args()

    #  DDP init 
    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically
    dist.init_process_group(
        backend="nccl" if not args.no_cuda and torch.cuda.is_available() else "gloo"
    )
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    #  Device
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #  Dirs (only rank 0 creates) 
    if rank == 0:
        os.makedirs(args.data_dir,    exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir,    exist_ok=True)

    dist.barrier()  # wait for rank 0 to finish creating dirs

    #  Logging 
    logger = setup_logging(rank, args.results_dir)

    if rank == 0:
        print("=" * 70)
        print("  DISTRIBUTED IMAGE RECOGNITION — PyTorch DDP + torchrun")
        print("=" * 70)
        print(f"  World size  : {world_size} process(es)")
        print(f"  Backend     : {'nccl' if device.type == 'cuda' else 'gloo'}")
        print(f"  Device      : {device}")
        print(f"  Host        : {socket.gethostname()}")
        print(f"  Epochs      : {args.epochs}")
        print(f"  Batch/GPU   : {args.batch_size}  (effective: {args.batch_size * world_size})")
        print(f"  LR          : {args.lr}")
        print("=" * 70)

    #  Data 
    train_loader, val_loader, train_sampler = build_dataloaders(args, rank, world_size)

    if rank == 0:
        print(f"\n  Train batches/rank : {len(train_loader)}")
        print(f"  Val   batches/rank : {len(val_loader)}")

    #  Model 
    model = build_model(args.num_classes).to(device)
    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        find_unused_parameters=False,   # faster; set True only if needed
    )

    #  Loss / Optimiser / Scheduler 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr * world_size,   # linear LR scaling rule
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Cosine annealing: smoothly decays LR to near-zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    # Mixed precision scaler (no-op on CPU)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    #  Resume 
    start_epoch = 0
    best_acc    = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        if rank == 0:
            print(f"\n  Resumed from {args.resume}  (epoch {start_epoch})")

    #  Metrics tracker 
    tracker = MetricsTracker()
    total_train_start = time.perf_counter()


    # TRAINING LOOP
 
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()

        # Must call set_epoch so shuffling differs per epoch across ranks
        train_sampler.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch, rank
        )

        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, rank
        )

        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)

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
            print(
                f"\n  Epoch {epoch:>3d}/{args.epochs-1} | "
                f"LR {current_lr:.5f} | "
                f"Train Loss {train_loss:.4f} | Acc@1 {train_acc1:.2f}% | "
                f"Val Loss {val_loss:.4f} | Acc@1 {val_acc1:.2f}% | "
                f"Best {best_acc:.2f}% | "
                f"Time {epoch_time:.1f}s"
                + (" ← best" if is_best else "")
            )

            # Save checkpoint
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

   
    # FINAL RESULTS  (rank 0 only)
    
    total_time = time.perf_counter() - total_train_start

    if rank == 0:
        final = {
            "mode"            : "distributed_ddp",
            "world_size"      : world_size,
            "backend"         : dist.get_backend(),
            "device"          : str(device),
            "hostname"        : socket.gethostname(),
            "epochs"          : args.epochs,
            "batch_per_gpu"   : args.batch_size,
            "effective_batch" : args.batch_size * world_size,
            "learning_rate"   : args.lr * world_size,
            "total_time_sec"  : round(total_time, 2),
            "avg_epoch_sec"   : round(total_time / args.epochs, 2),
            "best_val_acc1"   : round(best_acc, 2),
            "history"         : tracker.history,
        }

        out = os.path.join(args.results_dir, "distributed_results.json")
        with open(out, "w") as f:
            json.dump(final, f, indent=2)

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Total time   : {total_time:.2f}s")
        print(f"  Avg/epoch    : {total_time/args.epochs:.2f}s")
        print(f"  Best Val@1   : {best_acc:.2f}%")
        print(f"  Results      : {out}")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()