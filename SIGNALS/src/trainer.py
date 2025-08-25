
from enum import Enum
import json
import os
from typing import Any, Dict, Final, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from setting import APP_LOGGER, ColoredFormatter, TqdmLoggingHandler
from dataset.datamodule import Mitbih_datamodule
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import Logger

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


class Schedulers(Enum):
    """Enum per i tipi di scheduler di learning rate supportati."""
    STEP_LR = 'StepLR'
    EXPONENTIAL_LR = 'ExponentialLR'
    COSINE_ANNEALING_LR = 'CosineAnnealingLR'
    REDUCE_ON_PLATEAU = 'ReduceLROnPlateau'
    NONE = 'None'

class Trainer:
    
    __DATA_FILE_NAME: Final[str] = 'training_data.json'
    
    def __init__(
            self, 
            workingDirectory: str, 
            datamodule:Mitbih_datamodule, 
            device: torch.device, 
            model: torch.nn.Module, 
            top_k_checkpoints: int = 3
        ) -> None:
        """
        Inizializza il Trainer con un datamodule.
        
        Args:
            workingDirectory: Directory di lavoro
            datamodule: Oggetto datamodule che fornisce i dataloader.
            device: Dispositivo di calcolo (CPU/GPU)
            model: Modello da addestrare
            top_k_checkpoints: Numero di migliori checkpoint da mantenere
        """
        
        assert isinstance(workingDirectory, str), "workingDirectory must be a string"
        assert isinstance(datamodule, Mitbih_datamodule), "datamodule must be an instance of Mitbih_datamodule"
        assert isinstance(device, torch.device), "device must be an instance of torch.device"
        assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        assert isinstance(top_k_checkpoints, int) and top_k_checkpoints > 0, "top_k_checkpoints must be a positive integer"
        
        self._workingDirectory = os.path.join(workingDirectory, model.__class__.__name__)
        self._checkpointPath = os.path.join(self._workingDirectory, 'checkpoints')
        self._logsPath = os.path.join(self._workingDirectory, 'logs')
        self._pltPath = os.path.join(self._workingDirectory, 'plots')
        self._trainingDataPath = os.path.join(self._workingDirectory, 'training_data')
        self._dataModule = datamodule
        self._device = device
        self._model = model
        self._top_k_checkpoints = top_k_checkpoints
        
        self.__best_val_loss = float('inf')
        self.__start_epoch: int = 0
        self.__top_checkpoints: List[tuple] = []
        
    def __setupWorkingDirectory(self) -> bool:
        """Crea le directory necessarie per il training"""
        try:
            os.makedirs(self._checkpointPath, exist_ok=True)
            os.makedirs(self._logsPath, exist_ok=True)
            os.makedirs(self._pltPath, exist_ok=True)
            os.makedirs(self._trainingDataPath, exist_ok=True)
            return True
        except Exception as e:
            APP_LOGGER.error(f"Errore durante la creazione delle directory: {e}")
            return False
        
    def __restoreLastTrainingData(self) -> None:
        """Ripristina i dati dell'ultimo training"""
        path = os.path.join(self._trainingDataPath, Trainer.__DATA_FILE_NAME)
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.__best_val_loss = data.get('best_val_loss', float('inf'))
                    self.__start_epoch = data.get('start_epoch', 0)
                    self.__top_checkpoints = data.get('top_checkpoints', [])
                    APP_LOGGER.info(f"Ripristinati dati del training: epoca {self.__start_epoch}, best_val_loss {self.__best_val_loss}")
            except Exception as e:
                APP_LOGGER.error(f"Errore durante il ripristino dei dati di training: {e}")
        
        # Pulisci i checkpoint non necessari
        self.__cleanup_checkpoints()
        
        # Carica l'ultimo checkpoint se disponibile
        if self.__start_epoch > 0:
            self.__load_last_checkpoint()
    
    def __load_last_checkpoint(self) -> None:
        """Carica l'ultimo checkpoint disponibile"""
        if not self.__top_checkpoints:
            APP_LOGGER.warning("Nessun checkpoint trovato per il ripristino")
            return
            
        # Prendi il checkpoint con la loss di validazione più bassa
        best_checkpoint = min(self.__top_checkpoints, key=lambda x: x[0])
        checkpoint_path = best_checkpoint[1]
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(state_dict=checkpoint['model_state_dict'], strict=True)
            APP_LOGGER.info(f"Caricato checkpoint da {checkpoint_path}")
        except Exception as e:
            APP_LOGGER.error(f"Errore durante il caricamento del checkpoint: {e}")    
    
    def __cleanup_checkpoints(self) -> None:
        """Rimuove i checkpoint non necessari mantenendo solo i migliori top_k"""
        # Ottieni tutti i file nella directory dei checkpoint
        all_files = set(os.listdir(self._checkpointPath))
        
        # Mantieni solo i file dei checkpoint migliori
        checkpoint_files = set()
        for _, path in self.__top_checkpoints:
            if os.path.exists(path):
                filename = os.path.basename(path)
                checkpoint_files.add(filename)
        
        # Rimuovi i file non necessari
        for file in all_files:
            if file.endswith('.pt') and file not in checkpoint_files:
                try:
                    os.remove(os.path.join(self._checkpointPath, file))
                    APP_LOGGER.info(f"Rimosso checkpoint non necessario: {file}")
                except Exception as e:
                    APP_LOGGER.error(f"Errore durante la rimozione del checkpoint {file}: {e}")
    
        
    def __saveTrainingData(self) -> None:
        """Salva i dati di addestramento in un file JSON"""
        training_data = {
            'best_val_loss': self.__best_val_loss,
            'start_epoch': self.__start_epoch,
            'top_checkpoints': self.__top_checkpoints
        }
        
        path = os.path.join(self._trainingDataPath, Trainer.__DATA_FILE_NAME)
        
        try:
            with open(path, 'w') as f:
                json.dump(training_data, f, indent=4)
        except Exception as e:
            APP_LOGGER.error(f"Errore durante il salvataggio dei dati di training: {e}")
    
    def __save_checkpoint(self, epoch: int, val_loss: float, optimizer: torch.optim.Optimizer) -> None:
        """Salva un checkpoint e gestisce i migliori top_k checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.__best_val_loss
        }
        
        # Nome del file del checkpoint
        checkpoint_filename = f'checkpoint_epoch_{epoch}_val_loss_{val_loss:.4f}.pt'
        checkpoint_path = os.path.join(self._checkpointPath, checkpoint_filename)
        
        # Salva il checkpoint
        torch.save(checkpoint, checkpoint_path)
        APP_LOGGER.info(f"Salvato checkpoint: {checkpoint_filename}")
        
        # Aggiungi alla lista dei migliori checkpoint
        self.__top_checkpoints.append((val_loss, checkpoint_path))
        
        # Ordina per loss di validazione (crescente)
        self.__top_checkpoints.sort(key=lambda x: x[0])
        
        # Mantieni solo i top_k checkpoint
        if len(self.__top_checkpoints) > self._top_k_checkpoints:
            # Rimuovi il checkpoint con la loss più alta
            _, worst_path = self.__top_checkpoints.pop()
            try:
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    APP_LOGGER.info(f"Rimosso checkpoint con loss più alta: {os.path.basename(worst_path)}")
            except Exception as e:
                APP_LOGGER.error(f"Errore durante la rimozione del checkpoint: {e}")
        
        # Aggiorna i dati di training
        self.__saveTrainingData()
    

    
    def train(
            self, 
            num_epochs: int = 10, 
            lr: float = 0.001,
            schedulerType: Schedulers = Schedulers.NONE,
            scheduler_params: Dict[str, int | float] = {},
        ) -> None:
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
        
        # Inizializza lo scheduler se specificato
        scheduler_instance = None
        
        match schedulerType:
            
            case Schedulers.NONE:
                pass
            case Schedulers.STEP_LR:
                step_size = scheduler_params.get('step_size', 10)
                gamma = scheduler_params.get('gamma', 0.1)
                scheduler_instance = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            case Schedulers.EXPONENTIAL_LR:
                gamma = scheduler_params.get('gamma', 0.95)
                scheduler_instance = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            
            case Schedulers.COSINE_ANNEALING_LR:
                T_max = scheduler_params.get('T_max', num_epochs)
                eta_min = scheduler_params.get('eta_min', 0)
                scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            
            case Schedulers.REDUCE_ON_PLATEAU:
                mode = scheduler_params.get('mode', 'min')
                factor = scheduler_params.get('factor', 0.1)
                patience = scheduler_params.get('patience', 10)
                threshold = scheduler_params.get('threshold', 1e-4)
                scheduler_instance = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
                )
        
            
            
        
        class_weights = self._dataModule.get_train_dataset().get_class_weights()

        if class_weights is not None:
            class_weights = class_weights.to(self._device)
            criterion = FocalCrossEntropyLoss(alpha=1, gamma=2, reduction='mean', class_weights=class_weights)
        
        train_loader = self._dataModule.train_dataloader()
        val_loader = self._dataModule.val_dataloader()
        

        log_file_path = os.path.join(self._logsPath, 'training_log.csv')
        
        str1 = f"Epoch"
        str2 = f"Training"
        str3 = f"Validation"
        
        epoch_bar = tqdm(total=num_epochs, desc=f"{str1:<11}", position=0)
        training_bar = tqdm(total=len(train_loader), desc=f"{str2:<11}", position=1, unit="batch")
        validation_bar = tqdm(total=len(val_loader), desc=f"{str3:<11}", position=2, unit="batch")
    
        training_bar.set_postfix({"loss": f"{0:.4f}"})
        validation_bar.set_postfix({"loss": f"{0:.4f}"})
        epoch_bar.n = self.__start_epoch
        
        epoch_bar.refresh()
        training_bar.refresh()
        validation_bar.refresh()
        
        original_handlers = APP_LOGGER.handlers[:]   # copia lista handler
        
        try:
            for h in original_handlers:
                APP_LOGGER.removeHandler(h)
                
            tqdm_handler = TqdmLoggingHandler()
            console_formatter = ColoredFormatter(fmt=ColoredFormatter.FORMAT, datefmt=ColoredFormatter.DATE_FORMAT)
            tqdm_handler.setFormatter(console_formatter)
            APP_LOGGER.addHandler(tqdm_handler)
            APP_LOGGER.addHandler(original_handlers[-1])
   
   
            with open(log_file_path, 'a') as log_file:
                if os.path.getsize(log_file_path) == 0:
                    header = "Epoch,lr,Train_Loss,Val_Loss,Train_Acc,Val_Acc,Train_Precision_Macro,Val_Precision_Macro,Train_Recall_Macro,Val_Recall_Macro,Train_F1_Macro,Val_F1_Macro,Train_Precision_Micro,Val_Precision_Micro,Train_Recall_Micro,Val_Recall_Micro,Train_F1_Micro,Val_F1_Micro\n"
                    log_file.write(header)
                    log_file.flush()  # Assicurati che i dati vengano scritti immediatamente sul disco
                
                
            
                for epoch in range(self.__start_epoch, num_epochs):
                    APP_LOGGER.info(f"{'-'*80}")
                    APP_LOGGER.info(f"Epoca {epoch+1}/{num_epochs}")
                    
                    actual_lr = optimizer.param_groups[0]['lr']
                    epoch_bar.set_postfix({"lr": f"{actual_lr:.6f}"})
                    epoch_bar.refresh()
                    
                    ##### Training Step #####
                    self._model.train()
                    total_loss = 0.0
                    all_train_targets = []
                    all_train_predictions = []
                    training_bar.n = 0
                    training_bar.refresh()
                    
                    # Crea la barra di progresso per il training
                    # train_pbar = tqdm(
                    #     train_loader, 
                    #     total=len(train_loader),
                    #     desc=f"Training Epoch {epoch+1}/{num_epochs}",
                    #     unit="batch",
                    #     leave=True
                    # )
                    
                    for data in train_loader:
                        x = data['x2'].to(self._device)
                        y = data['y'].to(self._device)
                        
                        optimizer.zero_grad()
                        y_hat = self._model(x)
                        loss = criterion(y_hat, y)
                        loss.backward()
                        optimizer.step()
                        
                        # Aggiorna la loss totale
                        batch_loss = loss.item()
                        total_loss += batch_loss
                        
                        # Calcola le predizioni per le metriche
                        _, preds = torch.max(y_hat, 1)
                        all_train_targets.extend(y.cpu())
                        all_train_predictions.extend(preds.cpu())
                        
                        # Aggiorna la barra di progresso con la loss corrente
                        training_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                        training_bar.update(1) 
                        training_bar.refresh()
                    
                    # Chiudi la barra di progresso del training
                    #train_pbar.close()
                    
                    # Calcola le metriche di training
                    train_targets_tensor = torch.stack(all_train_targets)
                    train_preds_tensor = torch.stack(all_train_predictions)
                    train_metrics = self.__compute_metrics(train_targets_tensor, train_preds_tensor)
                    avg_train_loss = total_loss / len(train_loader)
                    
                    APP_LOGGER.info(f"Epoca {epoch+1} - Loss di training: {avg_train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
                    
                    
                    ##### Validation Step #####
                    self._model.eval()
                    val_loss = 0.0
                    all_val_targets = []
                    all_val_predictions = []
                    validation_bar.n = 0
                    validation_bar.refresh()
                    
                    # Crea la barra di progresso per la validazione
                    # val_pbar = tqdm(
                    #     val_loader,
                    #     total=len(val_loader),
                    #     desc=f"Validation Epoch {epoch+1}/{num_epochs}",
                    #     unit="batch",
                    #     leave=True
                    # )
                    
                    with torch.no_grad():
                        for data in val_loader:
                            x = data['x2'].to(self._device)
                            y = data['y'].to(self._device)
                            y_hat = self._model(x)
                            loss = criterion(y_hat, y)
                            
                            # Aggiorna la loss di validazione
                            batch_loss = loss.item()
                            val_loss += batch_loss
                            
                            # Calcola le predizioni per le metriche
                            _, preds = torch.max(y_hat, 1)
                            all_val_targets.extend(y.cpu())
                            all_val_predictions.extend(preds.cpu())
                            
                            # Aggiorna la barra di progresso con la loss corrente
                            validation_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                            validation_bar.update(1)  # Aggiorna manualmente la barra
                            validation_bar.refresh()
                    
                    # Chiudi la barra di progresso della validazione
                    #val_pbar.close()
                    
                    # Calcola le metriche di validazione
                    val_targets_tensor = torch.stack(all_val_targets)
                    val_preds_tensor = torch.stack(all_val_predictions)
                    val_metrics = self.__compute_metrics(val_targets_tensor, val_preds_tensor)
                    avg_val_loss = val_loss / len(val_loader)
                    
                    APP_LOGGER.info(f"Epoca {epoch+1} - Loss di validazione: {avg_val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                    
                    # Aggiorna lo scheduler
                    if scheduler_instance is not None:
                        match schedulerType:
                            case Schedulers.REDUCE_ON_PLATEAU:
                                scheduler_instance.step(avg_val_loss)
                            case _ :
                                scheduler_instance.step()
                                
                    # Salva il checkpoint se la loss di validazione è migliorata
                    if avg_val_loss < self.__best_val_loss:
                        self.__best_val_loss = avg_val_loss
                        self.__save_checkpoint(epoch, avg_val_loss, optimizer)
                        self.__save_confusion_matrix(val_targets_tensor, val_preds_tensor, epoch)
                    
                    # Aggiorna l'epoca di partenza per il prossimo training
                    self.__start_epoch = epoch + 1
                    self.__saveTrainingData()
                    
                    # Scrivi i risultati nel file di log
                    # Scrivi i risultati nel file di log
                    log_line = (
                        f"{epoch+1},{lr},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                        f"{train_metrics['accuracy']:.4f},{val_metrics['accuracy']:.4f},"
                        f"{train_metrics['precision_macro']:.4f},{val_metrics['precision_macro']:.4f},"
                        f"{train_metrics['recall_macro']:.4f},{val_metrics['recall_macro']:.4f},"
                        f"{train_metrics['f1_macro']:.4f},{val_metrics['f1_macro']:.4f},"
                        f"{train_metrics['precision_micro']:.4f},{val_metrics['precision_micro']:.4f},"
                        f"{train_metrics['recall_micro']:.4f},{val_metrics['recall_micro']:.4f},"
                        f"{train_metrics['f1_micro']:.4f},{val_metrics['f1_micro']:.4f}\n"
                    )
                    log_file.write(log_line)
                    log_file.flush()
                    epoch_bar.update(1)
                    epoch_bar.refresh()
        except Exception as e:
            APP_LOGGER.error(f"Errore durante l'addestramento: {e}") 
        finally:
            epoch_bar.close()
            training_bar.close()
            validation_bar.close()
            APP_LOGGER.info("Addestramento completato.") 
            
            for h in APP_LOGGER.handlers[:]:
                APP_LOGGER.removeHandler(h)   
            
            for h in original_handlers:
                APP_LOGGER.addHandler(h)
                   
    def __compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calcola le metriche di valutazione"""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true_np, y_pred_np),
            'precision_macro': precision_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true_np, y_pred_np, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true_np, y_pred_np, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        }
        
        return metrics

    def __save_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, epoch: int) -> None:
        """Salva la matrice di confusione come immagine"""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_true_np, y_pred_np)
        
        # Normalizza la matrice per visualizzazione migliore
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crea la figura
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            cbar=False
        )
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Salva l'immagine
        cm_path = os.path.join(self._pltPath, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        
        APP_LOGGER.info(f"Salvata matrice di confusione: {cm_path}")

        
            
