from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# BasicBlock1D come prima
class BasicBlock1D(nn.Module):
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


# Modello Generale - può fare classi o categorie a seconda dell'argomento
class ResNet1D_General(nn.Module):
    def __init__(self, block, layers, in_channels_signal=1, output_dim=1):
        super(ResNet1D_General, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels_signal, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        layers += [block(self.in_channels, out_channels) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)



# Modello ResNet50 Generale
class ResNet1D_50_General(nn.Module):
    def __init__(self, in_channels_signal=1, output_dim=1):
        super(ResNet1D_50_General, self).__init__()
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
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, output_dim)

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
        layers += [Bottleneck1D(self.in_channels, out_channels) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
                
                
# Versioni ResNet-18
class ResNet1D_18_Classes(ResNet1D_General):
    def __init__(self, in_channels_signal=1, classes_output_dim=1):
        super(ResNet1D_18_Classes, self).__init__(BasicBlock1D, [2, 2, 2, 2], in_channels_signal, classes_output_dim)

class ResNet1D_18_Categories(ResNet1D_General):
    def __init__(self, in_channels_signal=1, categories_output_dim=1):
        super(ResNet1D_18_Categories, self).__init__(BasicBlock1D, [2, 2, 2, 2], in_channels_signal, categories_output_dim)


# Versioni ResNet-34
class ResNet1D_34_Classes(ResNet1D_General):
    def __init__(self, in_channels_signal=1, classes_output_dim=1):
        super(ResNet1D_34_Classes, self).__init__(BasicBlock1D, [3, 4, 6, 3], in_channels_signal, classes_output_dim)

class ResNet1D_34_Categories(ResNet1D_General):
    def __init__(self, in_channels_signal=1, categories_output_dim=1):
        super(ResNet1D_34_Categories, self).__init__(BasicBlock1D, [3, 4, 6, 3], in_channels_signal, categories_output_dim)


# Versioni ResNet-50
class ResNet1D_50_Classes(ResNet1D_50_General):
    def __init__(self, in_channels_signal=1, classes_output_dim=1):
        super(ResNet1D_50_Classes, self).__init__(in_channels_signal, classes_output_dim)

class ResNet1D_50_Categories(ResNet1D_50_General):
    def __init__(self, in_channels_signal=1, categories_output_dim=1):
        super(ResNet1D_50_Categories, self).__init__(in_channels_signal, categories_output_dim)

