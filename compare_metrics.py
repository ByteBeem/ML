import json
import os

import numpy as np

RESULTS_DIR = "results"


def load(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)


def print_comparison(s, d):
    speedup    = s["total_time_sec"] / d["total_time_sec"]
    acc_delta  = abs(s["best_val_acc1"] - d["best_val_acc1"])
    epoch_speedup = s["avg_epoch_sec"] / d["avg_epoch_sec"]

    print("\n" + "=" * 72)
    print(f"  {'METRIC':<30}  {'SINGLE':>14}  {'DISTRIBUTED':>14}")
    print("=" * 72)

    rows = [
        ("Processes / GPUs",    "1",                          str(d["world_size"])),
        ("Effective batch size", str(s["batch_size"]),         str(d["effective_batch"])),
        ("Epochs",              str(s["epochs"]),              str(d["epochs"])),
        ("Total time (s)",      f"{s['total_time_sec']:.2f}", f"{d['total_time_sec']:.2f}"),
        ("Avg epoch time (s)",  f"{s['avg_epoch_sec']:.2f}",  f"{d['avg_epoch_sec']:.2f}"),
        ("Best Val Acc@1 (%)",  f"{s['best_val_acc1']:.2f}",  f"{d['best_val_acc1']:.2f}"),
    ]

    for label, sv, dv in rows:
        print(f"  {label:<30}  {sv:>14}  {dv:>14}")

    print("=" * 72)
    print(f"\n  Epoch speedup     : {epoch_speedup:.2f}×")
    print(f"  Total speedup     : {speedup:.2f}×")
    print(f"  Acc@1 difference  : {acc_delta:.2f}%")

    if speedup > 1.0:
        print(f"\n  ✓ DDP was {speedup:.2f}× faster with {d['world_size']} process(es).")
        ideal   = d["world_size"]
        eff     = speedup / ideal * 100
        print(f"    Ideal linear speedup would be {ideal:.1f}×  (efficiency: {eff:.0f}%)")
        print(f"    Overhead sources: NCCL all-reduce, data loading, LR scaling.")
    else:
        print(f"\n  ✗ Single machine was faster — world_size={d['world_size']} too small")
        print(f"    for this dataset/batch, or network latency dominates.")

    print()


def try_plot(s, d):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping charts.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Single Machine vs DDP Distributed — ResNet-50 on CIFAR-100",
        fontsize=14, fontweight="bold"
    )

    BLUE = "#4a90d9"
    RED  = "#e74c3c"

    s_hist = s["history"]
    d_hist = d["history"]

    s_epochs = [h["epoch"] for h in s_hist]
    d_epochs = [h["epoch"] for h in d_hist]

    #  1. Training loss 
    ax = axes[0][0]
    ax.plot(s_epochs, [h["train_loss"] for h in s_hist], color=BLUE, label="Single")
    ax.plot(d_epochs, [h["train_loss"] for h in d_hist], color=RED,  label="Distributed")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    #  2. Val Acc@1 
    ax = axes[0][1]
    ax.plot(s_epochs, [h["val_acc1"] for h in s_hist], color=BLUE, label="Single")
    ax.plot(d_epochs, [h["val_acc1"] for h in d_hist], color=RED,  label="Distributed")
    ax.set_title("Validation Accuracy @1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(True, alpha=0.3)

    #  3. Val Acc@5 
    ax = axes[0][2]
    ax.plot(s_epochs, [h["val_acc5"] for h in s_hist], color=BLUE, label="Single")
    ax.plot(d_epochs, [h["val_acc5"] for h in d_hist], color=RED,  label="Distributed")
    ax.set_title("Validation Accuracy @5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(True, alpha=0.3)

    #  4. Epoch time 
    ax = axes[1][0]
    ax.plot(s_epochs, [h["epoch_time"] for h in s_hist], color=BLUE, label="Single")
    ax.plot(d_epochs, [h["epoch_time"] for h in d_hist], color=RED,  label="Distributed")
    ax.set_title("Epoch Wall-Clock Time")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Seconds")
    ax.legend(); ax.grid(True, alpha=0.3)

    #  5. LR schedule 
    ax = axes[1][1]
    ax.plot(s_epochs, [h["lr"] for h in s_hist], color=BLUE, label="Single")
    ax.plot(d_epochs, [h["lr"] for h in d_hist], color=RED,  label="Distributed")
    ax.set_title("Learning Rate Schedule (Cosine)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.legend(); ax.grid(True, alpha=0.3)

    #  6. Summary bar
    ax = axes[1][2]
    categories = ["Total time (s)", "Avg epoch (s)", "Best Acc@1 (%)"]
    sv = [s["total_time_sec"], s["avg_epoch_sec"], s["best_val_acc1"]]
    dv = [d["total_time_sec"], d["avg_epoch_sec"], d["best_val_acc1"]]

    x     = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, sv, width, label="Single",      color=BLUE)
    ax.bar(x + width/2, dv, width, label="Distributed", color=RED)
    ax.set_title("Summary Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_charts.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Charts saved → {out}")
    plt.close()


def main():
    print("=" * 72)
    print("  METRICS COMPARISON — Single Machine vs DDP Distributed")
    print("=" * 72)

    missing = []
    for fname in ["single_machine_results.json", "distributed_results.json"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            missing.append(fname)
    if missing:
        print(f"\n  Missing: {missing}")
        print("  Run single_machine.py then train.py first.")
        return

    s = load("single_machine_results.json")
    d = load("distributed_results.json")

    print_comparison(s, d)
    try_plot(s, d)


if __name__ == "__main__":
    main()
