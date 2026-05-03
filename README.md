# Distributed Image Recognition — PyTorch DDP + torchrun

ResNet-50 trained on CIFAR-100 using PyTorch DistributedDataParallel.
Elapsed time **decreases** as you add more GPUs/machines because all
workers compute forward+backward passes **simultaneously** and sync
gradients in one efficient NCCL all-reduce.

---

## Project structure

```
ddp_project/
├── train.py            ← DDP training (launch with torchrun)
├── single_machine.py   ← Single GPU/CPU baseline
├── model.py            ← ResNet-50 adapted for 32×32 CIFAR images
├── utils.py            ← AverageMeter, accuracy, checkpointing
├── compare_metrics.py  ← Side-by-side comparison + charts
├── requirements.txt
├── data/               ← CIFAR-100 downloaded here automatically
├── checkpoints/        ← Saved model weights
├── results/            ← JSON metrics + PNG charts
└── logs/               ← Per-rank log files
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Run the single-machine baseline

```bash
# With GPU:
python single_machine.py

# CPU only:
python single_machine.py --no_cuda
```

Saves → `results/single_machine_results.json`

---

## Step 2 — Run distributed training

### Single machine, 1 GPU (same as baseline but with DDP overhead)
```bash
torchrun --standalone --nproc_per_node=1 train.py
```

### Single machine, 2 GPUs  ← first real speedup
```bash
torchrun --standalone --nproc_per_node=2 train.py
```

### Single machine, 4 GPUs
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

### Single machine, CPU only (for testing)
```bash
torchrun --standalone --nproc_per_node=1 train.py --no_cuda
```

Saves → `results/distributed_results.json`

---

## Adding more machines (scale out)

All machines must:
1. Have the same code (git clone or rsync)
2. Be able to reach each other on TCP port 29500
3. Have the same Python + PyTorch version installed

### 2 machines, 2 GPUs each (= 4 total processes)

**On machine 0 (master, e.g. 192.168.1.10):**
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train.py
```

**On machine 1:**
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --node_rank=1 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train.py
```

### 4 machines, 1 GPU each (= 4 total)

Run on each machine, changing only `--node_rank` (0, 1, 2, 3):
```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=1 \
  --node_rank=<0|1|2|3> \
  --master_addr=<machine0_ip> \
  --master_port=29500 \
  train.py
```

**Nothing else changes.** `DistributedSampler` automatically shards
the dataset across however many processes exist.

---

## Step 3 — Compare results

```bash
python compare_metrics.py
```

Prints a table like:

```
═══════════════════════════════════════════════════════════════════════════
  METRIC                           SINGLE    DISTRIBUTED
═══════════════════════════════════════════════════════════════════════════
  Processes / GPUs                      1              4
  Effective batch size                128            512
  Total time (s)                  3420.00          980.00
  Avg epoch time (s)               114.00           32.67
  Best Val Acc@1 (%)                62.40           63.10
═══════════════════════════════════════════════════════════════════════════

  Epoch speedup     : 3.49×
  Total speedup     : 3.49×
  Acc@1 difference  : 0.70%

  ✓ DDP was 3.49× faster with 4 process(es).
    Ideal linear speedup would be 4.0×  (efficiency: 87%)
```

Also saves `results/comparison_charts.png` with 6 plots.

---

## Why elapsed time DECREASES with DDP

```
Single machine:
  GPU 0: [fwd 50k samples] → [bwd] → [update]   ← sequential, full dataset

DDP with 4 GPUs:
  GPU 0: [fwd 12.5k] → [bwd] ──┐
  GPU 1: [fwd 12.5k] → [bwd] ──┤ all-reduce  → [update]
  GPU 2: [fwd 12.5k] → [bwd] ──┤ (overlapped
  GPU 3: [fwd 12.5k] → [bwd] ──┘  with bwd)
```

- Each GPU processes `1/world_size` of the data per epoch
- NCCL all-reduce runs **during** the backward pass (bucketed), hiding latency
- Result: ~linear speedup with efficiency typically 80–95%

---

## Resume training

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --resume checkpoints/checkpoint_epoch_009.pth
```

---

## Key hyperparameters

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 30 | Total training epochs |
| `--batch_size` | 128 | Per-GPU batch. Effective = batch × world_size |
| `--lr` | 0.1 | Base LR. Code applies linear scaling: lr × world_size |
| `--num_workers` | 4 | DataLoader workers per GPU |
| `--save_every` | 5 | Checkpoint every N epochs |

---

## Swapping to a larger dataset (ImageNet)

Replace the `build_dataloaders` function in `train.py`:

```python
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder("/path/to/imagenet/train", transform=train_transform)
val_dataset   = ImageFolder("/path/to/imagenet/val",   transform=val_transform)
```

Change `--num_classes 1000` and increase `--batch_size` and `--epochs`.
Everything else (DDP, sampler, all-reduce) stays identical.
