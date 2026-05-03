import torch.nn as nn


def build_model(num_classes: int = 100) -> nn.Module:
    """
    Lightweight linear classifier for CIFAR-100.
    Input: 32x32x3 = 3072 features (flattened)
    Output: num_classes logits

    Uses two small hidden layers so it's slightly more expressive
    than pure logistic regression but still tiny (~800KB of weights).
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [DEBUG] Model: lightweight MLP  |  parameters: {total:,}  |  "
          f"approx size: {total*4/1024/1024:.2f} MB")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    m = build_model(100)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    print(f"  Output shape : {y.shape}")
    print(f"  Parameters   : {count_parameters(m):,}")