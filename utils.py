import logging
import os
import shutil
import time
from typing import Tuple

import torch


# ─────────────────────────────────────────────
# AverageMeter
# ─────────────────────────────────────────────
class AverageMeter:
    """Computes and stores running average of a value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─────────────────────────────────────────────
# accuracy
# ─────────────────────────────────────────────
@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk: Tuple[int, ...] = (1,)) -> list:
    """Returns top-k accuracy (%) for each k in topk."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(float(correct_k * 100.0 / batch_size))
    return results


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def setup_logging(rank: int, base_dir: str = ".", debug: bool = False) -> logging.Logger:
    """
    Each rank writes its own log: <base_dir>/logs/rank_<rank>.log
    Rank 0 also echoes to stdout.
    Pass debug=True for DEBUG-level messages.
    """
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"rank_{rank}.log")

    level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers on reimport

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s][rank%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.debug(f"Logging initialised → {log_path}  (debug={debug})")
    return logger


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, is_best: bool,
                    ckpt_dir: str, epoch: int) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(state, fname)
    print(f"  [DEBUG] Checkpoint saved → {fname}")
    if is_best:
        best = os.path.join(ckpt_dir, "best.pth")
        shutil.copyfile(fname, best)
        print(f"  [DEBUG] Best checkpoint updated → {best}")


def load_checkpoint(path: str, device: torch.device) -> dict:
    print(f"  [DEBUG] Loading checkpoint from {path} onto {device}")
    return torch.load(path, map_location=device)


# ─────────────────────────────────────────────
# MetricsTracker
# ─────────────────────────────────────────────
class MetricsTracker:
    """Accumulates per-epoch metric dicts for JSON export."""

    def __init__(self):
        self.history = []

    def update(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch}
        entry.update(metrics)
        self.history.append(entry)

    def last(self) -> dict:
        return self.history[-1] if self.history else {}


# ─────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────
class Timer:
    """with Timer() as t: ...  → t.elapsed"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start