import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 100) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Adapt stem for small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    m = build_model(100)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"Output shape : {y.shape}")
    print(f"Parameters   : {count_parameters(m):,}")
