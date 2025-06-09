from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definizione del BasicBlock per input 1D
class BasicBlock1D(nn.Module):
    """
    Blocco residuo fondamentale per input 1D (segnali).
    """
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
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

# Definizione della ResNet per input 1D
class ResNet1D(nn.Module):
    """
    Implementazione della ResNet adattata per input 1D (segnali).
    """
    def __init__(self, in_channels_signal:int = 1, classes_output_dim: int = 1, categories_output_dim: int = 1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64 

        
        self.conv1 = nn.Conv1d(in_channels_signal, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Strati convoluzionali 1D con blocchi residui
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)


        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1_classes    = nn.Linear(512 * BasicBlock1D.expansion, classes_output_dim)
        self.fc2_categories = nn.Linear(512 * BasicBlock1D.expansion, categories_output_dim)

        # Inizializzazione dei pesi
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    def _make_layer(self, out_channels: int, blocks: int, stride: int | tuple):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * BasicBlock1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock1D.expansion),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock1D.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor] :
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
        x1 = self.fc1_classes(x)
        x2 = self.fc2_categories(x)

        return x1, x2
    
    
# Bottleneck block per ResNet-50 1D
class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ResNet-50 1D
class ResNet1D_50(nn.Module):
    def __init__(self, in_channels_signal=1, classes_output_dim=1, categories_output_dim=1):
        super(ResNet1D_50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels_signal, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512 * Bottleneck1D.expansion, classes_output_dim)
        self.fc2 = nn.Linear(512 * Bottleneck1D.expansion, categories_output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * Bottleneck1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * Bottleneck1D.expansion)
            )

        layers = [Bottleneck1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck1D.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        return x1, x2