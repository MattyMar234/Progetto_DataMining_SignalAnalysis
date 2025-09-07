import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_CNN_2D(nn.Module):
    def __init__(self):
        super(ECG_CNN_2D, self).__init__()
        
        # Primo blocco convoluzionale (8 filtri 4x4)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Secondo blocco convoluzionale (13 filtri 2x2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=13, kernel_size=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Terzo blocco convoluzionale (13 filtri 2x2)
        self.conv3 = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcolo dinamico delle dimensioni per il fully connected
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.conv3, self.relu3, self.pool3
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(13 * 30 * 30, 1024)  # 13*30*30 = 11700
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, 5)  # 5 classi di output
        
    def forward(self, x):
        # Passaggio attraverso i blocchi convoluzionali
        x = self.convs(x)
        
        # Appiattimento per i fully connected
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        x = x.view(-1, self._to_linear)
        
        # Passaggio attraverso i fully connected
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        
        return x
    
    
class ECG_CNN_1D(nn.Module):
    def __init__(self):
        super(ECG_CNN_1D, self).__init__()
        
        # Primo blocco convoluzionale (8 filtri di dimensione 4)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Secondo blocco convoluzionale (13 filtri di dimensione 2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=13, kernel_size=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Terzo blocco convoluzionale (13 filtri di dimensione 2)
        self.conv3 = nn.Conv1d(in_channels=13, out_channels=13, kernel_size=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calcolo dinamico delle dimensioni per il fully connected
        self._to_linear = None
        
        # Fully connected layers
        self.fc1 = nn.Linear(5824, 1024) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, 5)  # 5 classi di output
        
        
    def forward(self, x):
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Aggiunge dimensione canali
            
        # Passaggio attraverso i blocchi convoluzionali
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Appiattimento per i fully connected
        # if self._to_linear is None:
        #     self._to_linear = x[0].shape[0] * x[0].shape[1]
        #print(x.shape)
        x = x.view(-1, 5824)
        
        # Passaggio attraverso i fully connected
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        
        return x