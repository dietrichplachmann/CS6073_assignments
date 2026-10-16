"""Four models written from scratch for single-channel 28 x 28 images.

Every forward method returns ten unnormalized logits for CrossEntropyLoss.
VGG uses VGG11's convolution layout with a smaller MNIST classifier.
ResNet uses ResNet18's [2, 2, 2, 2] basic-block layout with a MNIST stem.
"""

import torch
from torch import nn


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for item in (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"):
            if item == "M":
                # Ceil pooling keeps a valid spatial dimension for 28px inputs.
                layers.append(nn.MaxPool2d(2, ceil_mode=True))
            else:
                layers.extend([nn.Conv2d(in_channels, item, 3, padding=1), nn.ReLU()])
                in_channels = item
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return torch.relu(self.residual(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        stages = []
        in_channels = 64
        for index, channels in enumerate((64, 128, 256, 512)):
            stages.append(BasicBlock(in_channels, channels, 1 if index == 0 else 2))
            stages.append(BasicBlock(channels, channels))
            in_channels = channels
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.stages(self.stem(x)))


MODEL_REGISTRY = {"DNN": DNN, "ConvNet": ConvNet, "VGG": VGG, "ResNet": ResNet}
