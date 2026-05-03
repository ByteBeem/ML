import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 100, arch: str = "resnet18") -> nn.Module:
    """
    arch choices: resnet18 (default, fits ~1GB RAM), resnet50 (needs 4GB+ RAM)
    """
    print(f"  [DEBUG] Building model: {arch} with {num_classes} classes")

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose resnet18 or resnet50.")

    # Adapt stem for small 32x32 CIFAR images (removes aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace classifier head for num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [DEBUG] Trainable parameters: {total_params:,}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    for arch in ["resnet18", "resnet50"]:
        m = build_model(100, arch=arch)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"  {arch} — output: {y.shape}  params: {count_parameters(m):,}")