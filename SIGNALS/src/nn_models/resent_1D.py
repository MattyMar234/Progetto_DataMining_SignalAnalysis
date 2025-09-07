import torch
import torch.nn as nn
from torch import Tensor

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(  # Modificato da Conv2d a Conv1d
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)  # Modificato da BatchNorm2d a BatchNorm1d
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(  # Modificato da Conv2d a Conv1d
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)  # Modificato da BatchNorm2d a BatchNorm1d
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(
        self,
        block,
        layers: list,
        input_channels: int = 1,
        num_classes: int = 1000
    ):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(  # Modificato da Conv2d a Conv1d
            input_channels, 
            64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)  # Modificato da BatchNorm2d a BatchNorm1d
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(  # Modificato da MaxPool2d a MaxPool1d
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Modificato da AdaptiveAvgPool2d a AdaptiveAvgPool1d
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Inizializzazione pesi modificata per Conv1d e BatchNorm1d
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(  # Modificato da Conv2d a Conv1d
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),  # Modificato da BatchNorm2d a BatchNorm1d
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        
        if x.dim() == 2:
            # Aggiungi dimensione canale: [batch, length] -> [batch, 1, length]
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18_1D(ResNet1D):
    def __init__(self, num_classes: int = 5):
        super().__init__(BasicBlock1D, [2, 2, 2, 2], 1, num_classes)
        
class ResNet34_1D(ResNet1D):
    def __init__(self, num_classes: int = 5):
        super().__init__(BasicBlock1D, [3, 4, 6, 3], 1, num_classes)