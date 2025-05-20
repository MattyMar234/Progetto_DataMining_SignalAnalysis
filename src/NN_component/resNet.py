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
        # Convoluzioni 1D
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion)
        self.downsample = downsample

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
    def __init__(self, in_channels_signal=1, output_dim=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64 # Numero iniziale di canali in uscita dal primo strato convoluzionale

        # Primo strato convoluzionale 1D
        # kernel_size=7, stride=2 per ridurre la lunghezza del segnale
        self.conv1 = nn.Conv1d(in_channels_signal, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # Max Pooling 1D
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Strati convoluzionali 1D con blocchi residui
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        # Global Average Pooling 1D
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Strato fully connected finale per regressione
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, output_dim)

        # Inizializzazione dei pesi
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten l'output per lo strato fully connected
        # x.shape sarà (batch_size, 512, 1) dopo avgpool, quindi flatten a (batch_size, 512)
        x = torch.flatten(x, 1)
        x = self.fc(x) # Output diretto per regressione

        return x