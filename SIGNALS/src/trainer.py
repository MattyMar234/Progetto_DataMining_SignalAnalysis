
import os
from typing import Any, Dict, Final, List
from dataset.datamodule import Mitbih_datamodule
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCrossEntropyLoss(nn.Module):
    """
    Implementazione della Focal Loss per problemi di classificazione con classi sbilanciate.
    Args:
        alpha (float): Peso per la classe positiva (default: 1)
        gamma (float): Fattore di focalizzazione (default: 2)
        reduction (str): Metodo di riduzione ('none', 'mean', 'sum')
    """
    def __init__(self, alpha:float | int = 1, gamma:float | int = 2, reduction:str='mean', class_weights: torch.Tensor | None=None, ignore_index: int = -100):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], "reduction must be one of ['none', 'mean', 'sum']"
        assert isinstance(alpha, (int, float)), "alpha must be a number"
        assert isinstance(gamma, (int, float)), "gamma must be a number"
        assert isinstance(ignore_index, int), "ignore_index must be an integer"
        
        if class_weights is not None:
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"
            assert class_weights.dim() == 1, "class_weights must be a 1D tensor"
        
        
        self._alpha: float | int = alpha
        self._gamma: float | int = gamma
        self._reduction: str = reduction
        
        self._criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='none'
        )
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        ce_loss = self._criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * ce_loss
        
        if self._reduction == 'mean':
            return focal_loss.mean()
        elif self._reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    
    __DATA_FILE_NAME: Final[str] = 'training_data.json'
    
    def __init__(self, workingDirectory: str, datamodule:Mitbih_datamodule, device: torch.device, model: torch.nn.Module):
        """
        Inizializza il Trainer con un datamodule.
        
        Args:
            datamodule: Oggetto datamodule che fornisce i dataloader.
        """
        
        assert isinstance(workingDirectory, str), "workingDirectory must be a string"
        assert isinstance(datamodule, Mitbih_datamodule), "datamodule must be an instance of Mitbih_datamodule"
        assert isinstance(device, torch.device), "device must be an instance of torch.device"
        assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        
        self._workingDirectory = os.path.join(workingDirectory, model.__class__.__name__)
        self._checkpointPath = os.path.join(self._workingDirectory, 'checkpoints')
        self._logsPath = os.path.join(self._workingDirectory, 'logs')
        self._pltPath = os.path.join(self._workingDirectory, 'plots')
        self._trainingDataPath = os.path.join(self._workingDirectory, 'training_data')
        self._dataModule = datamodule
        self._device = device
        self._model = model
        
        self.__best_val_loss = float('inf')
        self.__start_epoch: int = 0
        self.__top_checkpoints: List[str] = []
        
    def __setupWorkingDirectory(self) -> bool:
        
        try:
            os.makedirs(self._checkpointPath, exist_ok=True)
            os.makedirs(self._logsPath, exist_ok=True)
            os.makedirs(self._pltPath, exist_ok=True)
            os.makedirs(self._trainingDataPath, exist_ok=True)
            return True
        
        except Exception as e:
            print(e)
            return False
        
    def __restoreLastTrainingData(self) -> None:
        path = os.path.join(self._trainingDataPath, Trainer.__DATA_FILE_NAME)
        
        if os.path.exists(path):
            import json
            with open(path, 'r') as f:
                data = json.load(f)
                self.__best_val_loss = data.get('best_val_loss', float('inf'))
                self.__start_epoch = data.get('start_epoch', 0)
                self.__top_checkpoints = data.get('top_checkpoints', [])
        
        
    def __saveTrainingData(self) -> None:
        """
        Salva i dati di addestramento in un file JSON.
        """
        training_data = {
            'best_val_loss': self.__best_val_loss,
            'start_epoch': self.__start_epoch,
            'top_checkpoints': self.__top_checkpoints
        }
        
        path = os.path.join(self._trainingDataPath, Trainer.__DATA_FILE_NAME)
        
        with open(path, 'w') as f:
            import json
            json.dump(training_data, f, indent=4)
            
            

        
        
        
        
    
    def __validationStep(self, epoch: int, criterion: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
        pass
    
    def train(self, num_epochs: int = 10, lr: float = 0.001) -> Dict[str, Any]:
        """
        Addestra il modello utilizzando i dati di training.
        
        Args:
            model: Modello PyTorch da addestrare.
            epochs: Numero di epoche di addestramento (default: 10).
            lr: Learning rate per l'ottimizzatore (default: 0.001).
        """
        
        self.__setupWorkingDirectory()
        self.__restoreLastTrainingData()
        
        self._model.to(self._device)
        
        
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = FocalCrossEntropyLoss(alpha=1, gamma=2, reduction='mean')
        
        class_weights = self._dataModule.get_train_dataset().getClassWeights().to(self._device)


        
        train_loader = self._dataModule.train_dataloader()
        val_dataloader = self._dataModule.val_dataloader()
        
        with open(os.path.join(self._logsPath), 'a') as log_file:
            str_temp = f"{'*'}{' '*39} {self._model.__class__.__name__.upper()} TRAINING {' '*39}{'*'}\n"
            log_file.write(f"\n{'*'*len(str_temp)}\n")
            log_file.write(str_temp)
            log_file.write(f"{'*'*len(str_temp)}\n")
            log_file.write("Epoch, lr, Train_Loss, Val_Loss, Class_Accuracy, Class_F1_Macro, Class_Precision_Macro, Class_Recall_Macro, Category_Accuracy, Category_F1_Macro, Category_Precision_Macro, Category_Recall_Macro\n")
            log_file.flush()  # Assicurati che i dati vengano scritti immediatamente sul disco
            
        
            for epoch in range(self.__start_epoch, num_epochs):
                APP_LOGGER.info(f"Epoca {epoch+1}/{num_epochs}")

            
            self._model.train()
            
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self._device), y.to(self._device)
                optimizer.zero_grad()
                y_hat = self._model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate(self, model):
        """
        Valuta il modello utilizzando i dati di test.
        
        Args:
            model: Modello PyTorch da valutare.
            
        Returns:
            dict: Dizionario contenente loss e accuracy.
        """
        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        
        test_loader = self.datamodule.test_dataloader()
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }