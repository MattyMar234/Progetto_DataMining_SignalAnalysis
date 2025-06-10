import argparse
from enum import Enum, auto
import math
from typing import Final, List, Tuple
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, cohen_kappa_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from NN_component.VisionTransformer import ViT1D
from NN_component.VisionTransformer2 import ViT1D_2
from NN_component.resNet import *
from dataset.dataset import DatasetChannels, DatasetDataMode, DatasetMode, MITBIHDataset, BeatType
from dataset.datamodule import Mitbih_datamodule
import os

from NN_component.weightedMSELoss import WeightedMSELoss
from NN_component.model import SimpleECGRegressor, Transformer_BPM_Regressor
from setting import *
import setting


# Lista per tenere traccia dei migliori checkpoint (path e loss)
top_checkpoints: list = []

class TRAINING_MODE(Enum):
    CLASSES = auto()
    CATEGORIES = auto()


def check_pytorch_cuda() -> bool:
    #Globals.APP_LOGGER.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        setting.APP_LOGGER.info("CUDA is available on this system.")
        setting.APP_LOGGER.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            setting.APP_LOGGER.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        setting.APP_LOGGER.info("CUDA is not available on this system.")
        return False


def trainModel(
    device: torch.device, 
    dataModule: Mitbih_datamodule, 
    model: nn.Module, 
    num_epochs: int,
    start_lr: float = 1e-3,
    training_log_path:str | None = None,
    checkpoint_dir: str | None =  None,
    checkpoint: str | None = None
   ) -> None:
    
    assert training_log_path is not None, "File log di training non specificato"
    assert checkpoint_dir is not None, "Cartella dei checkpoint non specificata"
    
    # Assicurati che la directory esista
    os.makedirs(checkpoint_dir, exist_ok=True)

    
    start_epoch: int = 0
    best_val_loss = float('inf')
    
    model.to(device)
    train_dataloader = dataModule.train_dataloader()
    val_dataloader = dataModule.val_dataloader()
    # bins = dataModule.get_train_dataset().bpm_bins.to(device)
    # weights = dataModule.get_train_dataset().bin_weights.to(device)
    
    #funzione di loss
    #loss_function = WeightedMSELoss(bpm_bins=bins, weights=weights).to(device)
    loss_function = nn.MSELoss()
    
    #ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    #Inizializza lo scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Monitora una metrica da minimizzare (validation loss)
        factor=0.5,         # Fattore di riduzione del learning rate (LR = LR * factor)
        patience=3,         # Numero di epoche senza miglioramenti prima di ridurre il LR
        threshold=0.0001,   # Soglia per considerare un "miglioramento"
        threshold_mode='rel' # La soglia è relativa al valore corrente
    )
    
    
    if checkpoint is not None:
        # Carica un checkpoint esistente se presente
        if os.path.exists(checkpoint):
            APP_LOGGER.info(f"Caricamento checkpoint da {checkpoint}")
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            #best_val_loss = checkpoint_data['loss']
            start_epoch = checkpoint_data['epoch'] + 1
            APP_LOGGER.info(f"Riprendi l'addestramento dall'epoca {start_epoch} con validation loss {best_val_loss:.4f}")


    
    

    with open(training_log_path, 'a') as log_file:
        if os.stat(training_log_path).st_size != 0:
            log_file.write(f"\n{'='*80}\n")

        log_file.write("Epoch, lr, Train_Loss, Train_MAE, Train_RMSE, Val_Loss, Val_MAE, Val_RMSE\n")

        
        for epoch in range(start_epoch, num_epochs):
            APP_LOGGER.info(f"Epoca {epoch+1}/{num_epochs}")

            # --- Fase di Training ---
            model.train()
            total_train_loss = 0 # Accumula la loss per l'epoca
            total_train_mae = 0 # Accumula MAE per l'epoca
            
            # Utilizza tqdm per visualizzare l'avanzamento del training
            train_loop = tqdm(train_dataloader, leave=False, desc=f"Training Epoca {epoch+1}", )
            for batch_idx, (signal, bpm) in enumerate(train_loop):
                
                #bpm = torch.nn.functional.one_hot(bpm, 261).view(12, 261).float()
                signal = signal.to(device)
                bpm = bpm.float().squeeze(1).to(device) #da [12, 1] a [12]
                
                optimizer.zero_grad()
                outputs = model(signal).squeeze(1) #da [12, 1] a [12]
                
                #APP_LOGGER.info(bpm, outputs)

                # APP_LOGGER.info(f"bpm shape: {bpm.shape}")
                # APP_LOGGER.info(f"outputs shape: {outputs.shape}")


                # APP_LOGGER.info(outputs)
                # APP_LOGGER.info(outputs.shape)
                
                # # APP_LOGGER.info(bpm)
                # APP_LOGGER.info(bpm.shape)


                loss = loss_function(outputs, bpm)
                mae = torch.mean(torch.abs(outputs - bpm))
    

                loss.backward()
                optimizer.step()

                # total_train_loss += loss.item() * signal.size(0) # Moltiplica per la batch size per avere la somma reale
                # total_train_mae += mae.item() * signal.size(0) # Accumula MAE pesato per batch size
                total_train_loss += loss.item()
                total_train_mae += mae.item()
                
                # Aggiorna la descrizione di tqdm con la loss corrente
                train_loop.set_description(f"Training Epoca {epoch+1} Loss: {loss.item():.4f}")


            # Calcola le metriche medie per l'epoca di training
            avg_train_loss = total_train_loss / len(train_dataloader.dataset) # Divisione per il numero totale di campioni
            avg_train_mae = total_train_mae / len(train_dataloader.dataset)
            avg_train_rmse = math.sqrt(avg_train_loss) # RMSE è la radice quadrata dell'MSE medio


            # --- Fase di Validation ---
            model.eval()
            total_val_loss = 0 # Accumula la loss per l'epoca
            total_val_mae = 0 # Accumula MAE per l'epoca
            
            # Utilizza tqdm per visualizzare l'avanzamento della validation
            val_loop = tqdm(val_dataloader, leave=False, desc=f"Validation Epoca {epoch+1}")
            with torch.no_grad():
                for batch_idx, (signal, bpm) in enumerate(val_loop):
                    
                    #bpm = torch.nn.functional.one_hot(bpm, 261).view(12, 261).float()
                    signal = signal.to(device)
                    bpm = bpm.long().squeeze(1).to(device)
                    
                    #APP_LOGGER.info(bpm)
                    
                    outputs = model(signal).squeeze(1) #da [12, 1] a [12]
                    
                    # MSE Loss
                    loss = loss_function(outputs, bpm)

                    # Calcola MAE per il batch
                    mae = torch.mean(torch.abs(outputs - bpm.float()))
                    
                    total_val_loss += loss.item()
                    total_val_mae += mae.item()
                    # total_val_loss += loss.item() * signal.size(0) # Moltiplica per la batch size per avere la somma reale
                    # total_val_mae += mae.item() * signal.size(0) # Accumula MAE pesato per batch size
                    
                    # Aggiorna la descrizione di tqdm con la loss corrente
                    val_loop.set_description(f"Validation Epoca {epoch+1} Loss: {loss.item():.4f}")


             # Calcola le metriche medie per l'epoca di validation
            avg_val_loss = total_val_loss / len(val_dataloader.dataset) # Divisione per il numero totale di campioni
            avg_val_mae = total_val_mae / len(val_dataloader.dataset)
            avg_val_rmse = math.sqrt(avg_val_loss) # RMSE è la radice quadrata dell'MSE medio


            APP_LOGGER.info(f"Epoca {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

            # Applica lo step dello scheduler, passandogli la validation loss
            scheduler.step(avg_val_loss)

            # --- Log dei risultati ---
            log_file.write(f"{epoch+1}, {scheduler.get_last_lr()}, {avg_train_loss:.6f}, {avg_train_mae:.6f}, {avg_train_rmse:.6f}, {avg_val_loss:.6f}, {avg_val_mae:.6f}, {avg_val_rmse:.6f}\n")
            log_file.flush() # Assicurati che i dati vengano scritti immediatamente sul disco

            # log_file.write(f"{epoch+1}, {'-'}, {avg_train_loss:.6f}, {avg_train_mae:.6f}, {avg_train_rmse:.6f}, {avg_val_loss:.6f}, {avg_val_mae:.6f}, {avg_val_rmse:.6f}\n")
            # log_file.flush() # Assicurati che i dati vengano scritti immediatamente sul disco


            # --- Gestione Checkpoint ---
            # Salva il modello migliore basato sulla validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                # Definisci il percorso del file di checkpoint (salviamo solo il migliore)
                # Potresti volerne tenere di più o nominarli diversamente
                checkpoint_path = os.path.join(checkpoint_dir, f"Epoch[{epoch+1}]_Loss[{avg_val_loss:.4f}].pth")

                APP_LOGGER.info(f"Validation loss migliorata ({avg_val_loss:.4f}). Salvataggio modello in {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'loss': avg_val_loss, # Salva il miglior loss raggiunto
                }, checkpoint_path) # Salva nel file specificato





    APP_LOGGER.info("Addestramento completato.")

# def plot_confusion_matrix(cm, class_labels, epoch, output_dir):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
#     plt.title(f"Confusion Matrix - Epoch {epoch}")
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.tight_layout()
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))
#     plt.close()

def plot_confusion_matrix(cm, class_labels, epoch, output_dir, normalized=True, filename_prefix="confusion_matrix"):
    plt.figure(figsize=(12, 12))

    if normalized:
        # cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        # cm_normalized = np.nan_to_num(cm_normalized)  # evita NaN nelle divisioni per zero
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm.astype('float'), cm_sum, where=cm_sum != 0)
        cm_normalized = np.nan_to_num(cm_normalized)  # sostituisce eventuali NaN con 0

        
        annot = np.empty_like(cm, dtype=object)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                prob = cm_normalized[i, j]
                count = cm[i, j]
                annot[i, j] = f"{prob:.2f}\n({count})"
        
        sns.heatmap(cm_normalized, annot=annot, fmt='', cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)

    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_epoch_{epoch}.png"))
    plt.close()

def training_classification(
    training_mode: TRAINING_MODE,
    device: torch.device, 
    dataModule: Mitbih_datamodule, 
    model: nn.Module, 
    num_epochs: int,
    start_lr: float = 1e-3,
    training_log_path:str | None = None,
    checkpoint_dir: str | None =  None,
    confusion_matrix_dir: str | None = None,
    checkpoint: str | None = None
   ) -> None:
    
    assert training_log_path is not None, "File log di training non specificato"
    assert checkpoint_dir is not None, "Cartella dei checkpoint non specificata"
    assert confusion_matrix_dir is not None, "Cartella per le matrici di confusione non specificata"
    
    str_s = f"{'*'*40} TRAINING {'*'*40}"
    APP_LOGGER.info('*'*len(str_s))
    APP_LOGGER.info(str_s)
    APP_LOGGER.info('*'*len(str_s))
    
    confusion_matrix_dir_category = os.path.join(confusion_matrix_dir, 'category_ConfMatrix_plots')
    confusion_matrix_dir_class = os.path.join(confusion_matrix_dir, 'class_ConfMatrix_plots')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir_category, exist_ok=True)
    os.makedirs(confusion_matrix_dir_class, exist_ok=True)

    start_epoch: int = 0
    best_val_loss = float('inf')
    

    model.to(device)
    train_dataloader = dataModule.train_dataloader()
    val_dataloader = dataModule.val_dataloader()
    

    class_weights = dataModule.get_train_dataset().getClassWeights().to(device)
    class_to_ignore = BeatType.get_ignore_class_value()
    classes_number = BeatType.num_classes()
    
    category_weights = dataModule.get_train_dataset().getCategoryWeights().to(device)
    category_to_ignore = BeatType.get_ignore_category_value()
    categories_number = BeatType.num_of_category()
    

    class_loss_function = nn.CrossEntropyLoss(weight=class_weights, ignore_index=class_to_ignore)
    category_loss_function = nn.CrossEntropyLoss(weight=category_weights, ignore_index=category_to_ignore)
    
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',         
    #     factor=0.6,         
    #     patience=5,         
    #     threshold=0.0001,   
    #     threshold_mode='rel' 
    # )
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         
        factor=0.95,         
        patience=0,         
        threshold=0.0001,   
        threshold_mode='rel' 
    )
    
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=0, gamma=0.955)
    
    if checkpoint is not None:
        if os.path.exists(checkpoint):
            APP_LOGGER.info(f"Caricamento checkpoint da {checkpoint}")
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            start_epoch = checkpoint_data['epoch'] + 1
            APP_LOGGER.info(f"Riprendi l'addestramento dall'epoca {start_epoch} con validation loss {best_val_loss:.4f}")

    

    with open(training_log_path, 'a') as log_file:
        if os.stat(training_log_path).st_size != 0:
            log_file.write(f"\n{'='*80}\n")

        log_file.write("Epoch, lr, Train_Loss, Val_Loss, Class_Accuracy, Class_F1_Macro, Class_Precision_Macro, Class_Recall_Macro, Category_Accuracy, Category_F1_Macro, Category_Precision_Macro, Category_Recall_Macro\n")
        
        for epoch in range(start_epoch, num_epochs):
            APP_LOGGER.info(f"Epoca {epoch+1}/{num_epochs}")

            # --- Fase di Training ---
            total_train_loss: float = 0.0
            total_val_loss:float = 0.0
            
            train_class_preds = []
            train_class_targets = []
            train_category_preds = []
            train_category_targets = []
            
            val_class_preds = []
            val_class_targets = []
            val_category_preds = []
            val_category_targets = []

            APP_LOGGER.info("Training in modalità Categorie")
            model.train()
            train_loop = tqdm(train_dataloader, leave=False, desc=f"Training Epoca {epoch+1}")
            
            if training_mode == TRAINING_MODE.CATEGORIES:
                for batch_idx, (signal, class_labels, category_labels) in enumerate(train_loop):
                    signal = signal.to(device)
                    category_labels = category_labels.squeeze(1).long().to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(signal) # Outputs are logits
                    
                    loss = category_loss_function(outputs, category_labels) # Assuming model outputs can be used for category directly or needs adjustment
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    
                    # Collect predictions and true labels for classes
                    category_preds = torch.argmax(outputs, dim=1)
                    valid_category_indices = (category_labels != category_to_ignore)
                    train_category_preds.extend(category_preds[valid_category_indices].cpu().numpy())
                    train_category_targets.extend(category_labels[valid_category_indices].cpu().numpy())
                    train_loop.set_description(f"Training Epoca {epoch+1} Loss: {loss.item():.4f} lr: {scheduler.get_last_lr()[0]:.8f}")

            elif training_mode == TRAINING_MODE.CLASSES:
                for batch_idx, (signal, class_labels, category_labels) in enumerate(train_loop):
                    signal = signal.to(device)
                    class_labels = class_labels.squeeze(1).long().to(device)
            
                    optimizer.zero_grad()
                    outputs = model(signal)
                    loss = class_loss_function(outputs, class_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                    # Collect predictions and true labels for classes
                    class_preds = torch.argmax(outputs, dim=1)
                    valid_class_indices = (class_labels != class_to_ignore)
                    train_class_preds.extend(class_preds[valid_class_indices].cpu().numpy())
                    train_class_targets.extend(class_labels[valid_class_indices].cpu().numpy())
                    train_loop.set_description(f"Training Epoca {epoch+1} Loss: {loss.item():.4f} lr: {scheduler.get_last_lr()[0]:.8f}")
            else: 
                raise ValueError("training_mode deve essere TRAINING_MODE.CLASSES o TRAINING_MODE.CATEGORIES")
            
            
            avg_train_loss = total_train_loss / len(train_dataloader) # Divide by number of batches

            # --- Fase di Validation ---
            model.eval()
            
            val_loop = tqdm(val_dataloader, leave=False, desc=f"Validation Epoca {epoch+1}")
            with torch.no_grad():
                
                if training_mode == TRAINING_MODE.CATEGORIES:  
                    for batch_idx, (signal, class_labels, category_labels) in enumerate(val_loop):
                        signal = signal.to(device)
                        category_labels = category_labels.squeeze(1).long().to(device)
                        outputs = model(signal)
                        
                        loss = category_loss_function(outputs, category_labels)
                        total_val_loss += loss.item()
                        
                        if math.isnan(loss):
                            print(loss, loss, total_val_loss, category_labels, torch.argmax(outputs[1], dim=1))
                        
                        # Collect predictions and true labels for categories
                        category_preds  = torch.argmax(outputs, dim=1)
                        valid_category_indices = (category_labels != category_to_ignore)
                        val_category_preds.extend(category_preds[valid_category_indices].cpu().numpy())
                        val_category_targets.extend(category_labels[valid_category_indices].cpu().numpy())
                        
                        val_loop.set_description(f"Validation Epoca {epoch+1} Loss: {loss.item():.4f}")

                elif training_mode == TRAINING_MODE.CLASSES:
                    for batch_idx, (signal, class_labels, category_labels) in enumerate(val_loop):
                        signal = signal.to(device)
                        class_labels = class_labels.squeeze(1).long().to(device)
                        outputs = model(signal)
                        
                        loss = class_loss_function(outputs, class_labels)
                        total_val_loss += loss.item()
                        
                        if math.isnan(loss):
                            print(loss, loss, total_val_loss, category_labels, torch.argmax(outputs[1], dim=1))
                        
                        # Collect predictions and true labels for classes
                        class_preds = torch.argmax(outputs, dim=1)
                        valid_class_indices = (class_labels != class_to_ignore)
                        val_class_preds.extend(class_preds[valid_class_indices].cpu().numpy())
                        val_class_targets.extend(class_labels[valid_class_indices].cpu().numpy())

                        val_loop.set_description(f"Validation Epoca {epoch+1} Loss: {loss.item():.4f}")



            avg_val_loss = total_val_loss / len(val_dataloader) # Divide by number of batches


            APP_LOGGER.info('-'*100)
            APP_LOGGER.info(f"Risultati Epoca {epoch+1}: ")
            APP_LOGGER.info(f"Training Loss = {avg_train_loss:.4f}")
            APP_LOGGER.info(f"Validation Loss = {avg_val_loss:.4f}")
            
            # Calculate metrics for Classes
            if training_mode == TRAINING_MODE.CLASSES:
                unique_class_labels_for_metrics = [i for i in range(classes_number) if i != class_to_ignore]
                cm_class = confusion_matrix(val_class_targets, val_class_preds, labels=unique_class_labels_for_metrics)
                #plot_confusion_matrix(cm_class, [BeatType.mapBeatClass_to_Label(i) for i in unique_class_labels_for_metrics], epoch + 1, confusion_matrix_dir)

                accuracy_class = accuracy_score(val_class_targets, val_class_preds)
                f1_macro_class = f1_score(val_class_targets, val_class_preds, average='macro', labels=unique_class_labels_for_metrics, zero_division=0)
                precision_macro_class = precision_score(val_class_targets, val_class_preds, average='macro', labels=unique_class_labels_for_metrics, zero_division=0)
                recall_macro_class = f1_score(val_class_targets, val_class_preds, average='macro', labels=unique_class_labels_for_metrics, zero_division=0) 

                log_file.write(f"{epoch+1}, {optimizer.param_groups[0]['lr']:.6f}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {accuracy_class:.6f}, {f1_macro_class:.6f}, {precision_macro_class:.6f}, {recall_macro_class:.6f}, -, -, -, -\n")
                log_file.flush()
                APP_LOGGER.info(f"Class Metrics    - Accuracy: {accuracy_class:.4f}, F1-Macro: {f1_macro_class:.4f}, Precision-Macro: {precision_macro_class:.4f}, Recall-Macro: {recall_macro_class:.4f}")
            

            else:
                # Calculate metrics for Categories
                unique_category_labels_for_metrics = [i for i in range(categories_number) if i != category_to_ignore]
                cm_category = confusion_matrix(val_category_targets, val_category_preds, labels=unique_category_labels_for_metrics)
                #plot_confusion_matrix(cm_category, [BeatType.mapBeatCategory_to_Label(i) for i in unique_category_labels_for_metrics], epoch + 1, os.path.join(confusion_matrix_dir, "category_confusion_matrix")) # Save category CM in a subfolder or with a different name

                accuracy_category = accuracy_score(val_category_targets, val_category_preds)
                f1_macro_category = f1_score(val_category_targets, val_category_preds, average='macro', labels=unique_category_labels_for_metrics, zero_division=0)
                precision_macro_category = precision_score(val_category_targets, val_category_preds, average='macro', labels=unique_category_labels_for_metrics, zero_division=0)
                recall_macro_category = f1_score(val_category_targets, val_category_preds, average='macro', labels=unique_category_labels_for_metrics, zero_division=0)
            
                log_file.write(f"{epoch+1}, {optimizer.param_groups[0]['lr']:.6f}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, -, -, -, -, {accuracy_category:.6f}, {f1_macro_category:.6f}, {precision_macro_category:.6f}, {recall_macro_category:.6f}\n")
                log_file.flush()
                APP_LOGGER.info(f"Category Metrics - Accuracy: {accuracy_category:.4f}, F1-Macro: {f1_macro_category:.4f}, Precision-Macro: {precision_macro_category:.4f}, Recall-Macro: {recall_macro_category:.4f}")
            

            if ((epoch+1) % 5 == 0 and epoch != 0) or (avg_val_loss < best_val_loss):

                if training_mode == TRAINING_MODE.CLASSES:
                    plot_confusion_matrix(
                        cm_class,
                        [BeatType.mapBeatClass_to_Label(i) for i in unique_class_labels_for_metrics],
                        epoch + 1,
                        confusion_matrix_dir_class,
                        normalized=True,
                        filename_prefix=f"epoch_{epoch+1}_class_confusion_matrix"
                    )
                if training_mode == TRAINING_MODE.CATEGORIES:
                    plot_confusion_matrix(
                        cm_category,
                        [BeatType.mapBeatCategory_to_Label(i) for i in unique_category_labels_for_metrics],
                        epoch + 1,
                        confusion_matrix_dir_category,
                        normalized=True,
                        filename_prefix=f"epoch_{epoch+1}_category_confusion_matrix"
                    )
            APP_LOGGER.info('-'*100)
            

            scheduler.step(avg_val_loss)
            #scheduler.step(epoch)

            # --- Gestione Checkpoint ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # checkpoint_path = os.path.join(checkpoint_dir, f"Epoch[{epoch+1}]_Loss[{avg_val_loss:.4f}].pth")
                # APP_LOGGER.info(f"Validation loss migliorata ({avg_val_loss:.4f}). Salvataggio modello in {checkpoint_path}")
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(), 
                #     'loss': avg_val_loss,
                # }, checkpoint_path)
                APP_LOGGER.info(f"Validation loss migliorata ({avg_val_loss:.4f}). Salvataggio checkpoint...")
                top_checkpoints = save_top_checkpoints(
                    avg_val_loss, epoch, model, optimizer, scheduler, checkpoint_dir, top_k=3
                )

    APP_LOGGER.info("Addestramento completato.")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"Epoch[{epoch+1}]_Loss[{avg_val_loss:.4f}].pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 
        'loss': avg_val_loss,
    }, checkpoint_path)

def save_top_checkpoints(avg_val_loss, epoch, model, optimizer, scheduler, checkpoint_dir, top_k=3):
    global top_checkpoints  # Per mantenere stato tra epoche

    checkpoint_path = os.path.join(
        checkpoint_dir, f"Epoch[{epoch+1}]_Loss[{avg_val_loss:.4f}].pth"
    )

    # Salvataggio del nuovo checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_val_loss,
    }, checkpoint_path)

    # Aggiungi il nuovo checkpoint alla lista
    top_checkpoints.append((avg_val_loss, checkpoint_path))

    # Ordina la lista in base alla loss (minore è meglio)
    top_checkpoints = sorted(top_checkpoints, key=lambda x: x[0])

    # Se abbiamo più di top_k checkpoint, rimuovi quelli peggiori
    if len(top_checkpoints) > top_k:
        worst_checkpoint = top_checkpoints.pop(-1)  # Peggiore (ultimo)
        if os.path.exists(worst_checkpoint[1]):
            os.remove(worst_checkpoint[1])  # Elimina file dal disco

    return top_checkpoints  # opzionale, per debug/logging


def test_model(
    device: torch.device, 
    datamodule: Mitbih_datamodule, 
    model: nn.Module, 
    checkpoint: str,
    task_type: str = "regression" # Add task_type to differentiate
    ):
    
    model.to(device)
    
    if task_type == "regression":
        loss_function = nn.MSELoss()
    elif task_type == "classification":
        # For testing classification, we might not need weights, but ignore_index is crucial
        loss_function = nn.CrossEntropyLoss(ignore_index=6) 
    else:
        raise ValueError("Invalid task_type for test_model. Must be 'regression' or 'classification'.")

    model.eval()

    total_test_loss = 0  # Accumula la loss per l'intero set di test
    total_test_mae = 0   # Accumula MAE per l'intero set di test
    num_samples = 0      # Conta il numero totale di campioni elaborati

   
    APP_LOGGER.info(f"Caricamento checkpoint da {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
   
    test_dataloader = datamodule.test_dataloader()
    batch_size = test_dataloader.batch_size
    # Disabilita il calcolo dei gradienti durante la valutazione
    
    APP_LOGGER.info("Avvio valutazione sul set di test...")
    with torch.no_grad():
        # Utilizza tqdm per visualizzare l'avanzamento della valutazione
        #test_loop = tqdm(test_dataloader, desc="Valutazione Test Set")

        for batch_idx, (signal, bpm) in enumerate(test_dataloader):
 
            # Sposta i dati e le etichette sul dispositivo
            signal = signal.to(device)
            # Assicurati che le etichette target siano float e sul dispositivo
            bpm = bpm.to(device).float()

            # Forward pass
            outputs = model(signal)
            
            

            # Calcola la loss del batch (per l'accumulo)
            #loss = loss_function(outputs, bpm)
            
            for i in range(batch_size):
                APP_LOGGER.info(f"Target: {bpm[i]} Predected: {outputs[i]} Loss: -")

            print()

            # Calcola MAE per il batch
            mae = torch.mean(torch.abs(outputs - bpm))

            # Accumula le metriche, pesate per la dimensione del batch corrente
            batch_size = signal.size(0)
            #total_test_loss += loss.item() * batch_size
            total_test_mae += mae.item() * batch_size
            num_samples += batch_size

            # Aggiorna la descrizione di tqdm con la loss corrente del batch
            #test_loop.set_description(f"Valutazione Test Set Loss: {loss.item():.4f}")


    # Calcola le metriche medie sull'intero set di test
    avg_test_loss = total_test_loss / num_samples
    avg_test_mae = total_test_mae / num_samples
    avg_test_rmse = math.sqrt(avg_test_loss) # RMSE è la radice quadrata dell'MSE medio

    APP_LOGGER.info("\n--- Risultati Test Set ---")
    APP_LOGGER.info(f"Test Loss (MSE): {avg_test_loss:.6f}")
    APP_LOGGER.info(f"Test MAE: {avg_test_mae:.6f}")
    APP_LOGGER.info(f"Test RMSE: {avg_test_rmse:.6f}")
    APP_LOGGER.info("--------------------------")

    metrics = {
        'test_loss': avg_test_loss,
        'test_mae': avg_test_mae,
        'test_rmse': avg_test_rmse
    }

    return metrics
  
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_labels: list,
    category_labels: list,
    class_to_ignore: int,
    category_to_ignore: int,
    class_label_mapper,
    category_label_mapper,
    output_dir: str,
    name: str = "eval"
):
    model.eval()
    model.to(device)

    all_class_preds = []
    all_class_targets = []
    all_category_preds = []
    all_category_targets = []

    with torch.no_grad():
        for signals, class_targets, category_targets in tqdm(dataloader, desc=f"Evaluating {name}"):
            signals = signals.to(device)
            class_targets = class_targets.squeeze(1).long().to(device)
            category_targets = category_targets.squeeze(1).long().to(device)

            outputs = model(signals)
            class_preds = torch.argmax(outputs[0], dim=1)
            category_preds = torch.argmax(outputs[1], dim=1)

            valid_class_mask = (class_targets != class_to_ignore)
            valid_category_mask = (category_targets != category_to_ignore)

            all_class_preds.extend(class_preds[valid_class_mask].cpu().numpy())
            all_class_targets.extend(class_targets[valid_class_mask].cpu().numpy())

            all_category_preds.extend(category_preds[valid_category_mask].cpu().numpy())
            all_category_targets.extend(category_targets[valid_category_mask].cpu().numpy())

    # CLASS METRICS
    unique_class_labels = [i for i in range(len(class_labels)) if i != class_to_ignore]
    cm_class = confusion_matrix(all_class_targets, all_class_preds, labels=unique_class_labels)
    plot_confusion_matrix(
        cm_class,
        [class_label_mapper(i) for i in unique_class_labels],
        f"Confusion Matrix - Classes ({name})",
        os.path.join(output_dir, f"confusion_matrix_class_{name}.png")
    )

    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        all_class_targets, all_class_preds, labels=unique_class_labels, average=None, zero_division=0
    )
    acc_c = accuracy_score(all_class_targets, all_class_preds)
    kappa_c = cohen_kappa_score(all_class_targets, all_class_preds)

    weighted_avg_class = plot_confusion_matrix(
        all_class_targets, all_class_preds, labels=unique_class_labels, average="weighted", zero_division=0
    )

    # CATEGORY METRICS
    unique_category_labels = [i for i in range(len(category_labels)) if i != category_to_ignore]
    cm_cat = confusion_matrix(all_category_targets, all_category_preds, labels=unique_category_labels)
    plot_confusion_matrix(
        cm_cat,
        [category_label_mapper(i) for i in unique_category_labels],
        f"Confusion Matrix - Categories ({name})",
        os.path.join(output_dir, f"confusion_matrix_category_{name}.png")
    )

    precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
        all_category_targets, all_category_preds, labels=unique_category_labels, average=None, zero_division=0
    )
    acc_t = accuracy_score(all_category_targets, all_category_preds)
    kappa_t = cohen_kappa_score(all_category_targets, all_category_preds)

    weighted_avg_category = precision_recall_fscore_support(
        all_category_targets, all_category_preds, labels=unique_category_labels, average="weighted", zero_division=0
    )

    # Report
    APP_LOGGER.info(f"\n--- EVALUATION REPORT: {name.upper()} ---")
    APP_LOGGER.info(f"\n-> CLASS RESULTS:")
    for i, idx in enumerate(unique_class_labels):
        APP_LOGGER.info(f"Class {class_label_mapper(idx)}: Precision={precision_c[i]:.4f}, Recall={recall_c[i]:.4f}, F1={f1_c[i]:.4f}")
    APP_LOGGER.info(f"Weighted Avg (Class): Precision={weighted_avg_class[0]:.4f}, Recall={weighted_avg_class[1]:.4f}, F1={weighted_avg_class[2]:.4f}")
    APP_LOGGER.info(f"Overall Accuracy (Class): {acc_c:.4f}, Kappa: {kappa_c:.4f}")

    APP_LOGGER.info(f"\n-> CATEGORY RESULTS:")
    for i, idx in enumerate(unique_category_labels):
        APP_LOGGER.info(f"Category {category_label_mapper(idx)}: Precision={precision_t[i]:.4f}, Recall={recall_t[i]:.4f}, F1={f1_t[i]:.4f}")
    APP_LOGGER.info(f"Weighted Avg (Category): Precision={weighted_avg_category[0]:.4f}, Recall={weighted_avg_category[1]:.4f}, F1={weighted_avg_category[2]:.4f}")
    APP_LOGGER.info(f"Overall Accuracy (Category): {acc_t:.4f}, Kappa: {kappa_t:.4f}")


def main():

    parser = argparse.ArgumentParser(description="Script per configurare e addestrare un modello Transformer sui dati MIT-BIH.")
    
    # Parametri per la struttura del modello Transformer
   

    # parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help=f"Numero di strati del Transformer (default: {NUM_LAYERS})")
    # parser.add_argument("--d_model", type=int, default=D_MODEL, help=f"Dimensione del modello (default: {D_MODEL})")
    # parser.add_argument("--num_heads", type=int, default=NUM_HEADS, help=f"Numero di teste di attenzione (default: {NUM_HEADS})")
    # parser.add_argument("--dff", type=int, default=DFF, help=f"Dimensione del feed-forward (default: {DFF})")
    # parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE, help=f"Tasso di dropout (default: {DROPOUT_RATE})")
    
    # Percorsi e configurazioni
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help=f"Percorso del dataset (default: {DATASET_PATH})")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Percorso di un addestramento precedente (default: None)")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help=f"Percorso di output per i risultati (default: {OUTPUT_PATH})")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help=f"Numero di epoche di training (default: {NUM_EPOCHS})")
    parser.add_argument("--mode", type=str, choices=["training", "test"], default="training", help="Modalità di esecuzione: 'training' o 'test' (default: 'training')")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help=f"Valore iniziale del learning rate (default: {LEARNING_RATE})")

    # Configurazioni del segnale
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help=f"Dimensione della finestra del segnale in secondi (default: {WINDOW_SIZE})")
    parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE, help=f"Stride della finestra del segnale in secondi (default: {WINDOW_STRIDE})")
    args = parser.parse_args()
    

    
    
    # Calcola i parametri del dataset in base alla frequenza di campionamento
    sample_rate = 360
    sample_per_window = sample_rate * args.window_size
    sample_per_side = sample_rate * args.window_stride
    channels_enum = DatasetChannels.TWO
    
    sample_per_window = 280 #int(sample_rate*0.6),#args.window_size,
    
    
    models: List[Tuple[nn.Module, TRAINING_MODE]] = [
    
        (
            ResNet1D_18_Categories(
                in_channels_signal=channels_enum.value,
                categories_output_dim=BeatType.num_of_category()
            ),
            TRAINING_MODE.CATEGORIES
        ),
        (
            ResNet1D_18_Classes(
                in_channels_signal=channels_enum.value,
                classes_output_dim=BeatType.num_classes()
            ),
            TRAINING_MODE.CLASSES  
        )
        

        # ViT1D(
        #     signal_length = sample_per_window,
        #     patch_size = 12,
        #     in_channels=channels_enum.value,
        #     emb_dim=150,
        #     depth=10,
        #     num_heads=10,
        #     mlp_dim=150*3,
        #     classes_output_dim=BeatType.num_classes(),
        #     categories_output_dim=BeatType.num_of_category(),
        #     dropout = 0.3
        # )
        
        # ViT1D_2(
        #     signal_length = sample_per_window,
        #     in_channels=channels_enum.value,
        #     emb_dim=128,
        #     depth=10,
        #     num_heads=16,
        #     mlp_dim=128*4,
        #     classes_output_dim=BeatType.num_classes(),
        #     categories_output_dim=BeatType.num_of_category(),
        #     dropout = 0.3
        # )
    ]
    
    dataModule = Mitbih_datamodule(
        args.dataset_path, 
        datasetDataMode=DatasetDataMode.BEAT_CLASSIFICATION,
        datasetChannels=channels_enum,
        sample_rate=sample_rate, 
        sample_per_window=sample_per_window,
        num_workers=4,
        batch_size=128*(2 // channels_enum.value)#12
    )
    
    

    # model = Transformer_BPM_Regressor(
    #     input_samples_num=args.window_size*sample_rate,
    #     in_channels=2,
    #     conv_kernel_size=200,
    #     conv_stride=200,
    #     d_model=args.d_model,
    #     head_num=8,
    #     num_encoder_layers=8,
    #     dim_feedforward=args.dff,
    #     dropout=args.dropout_rate,
    # )
    
    # model = SimpleECGRegressor(
    #     in_channels=2,
    #     input_length=args.window_size*sample_rate
    # )
    
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    APP_LOGGER.info(f"Utilizzando dispositivo: {device}")
    
    for (model, mode) in models:
        # APP_LOGGER.info("Architettura del Modello:")
        APP_LOGGER.info(f"Training del Modello {model.__class__.__name__} per {mode}")
        
        training_classification(
            training_mode= mode,
            device = device,
            dataModule = dataModule,
            model = model,
            num_epochs = args.num_epochs,
            training_log_path=os.path.join(setting.LOGS_FOLDER, 'training_logs.txt'),
            checkpoint_dir = os.path.join(setting.OUTPUT_PATH, model.__class__.__name__),
            confusion_matrix_dir=os.path.join(setting.OUTPUT_PATH, model.__class__.__name__)#setting.LOGS_FOLDER
            #checkpoint = "/app/Data/Models/Epoch[1]_Loss[inf].pth"
        
        )


    
    # test_model(
    #     device = device,
    #     datamodule = dataModule,
    #     model = model,
    #     checkpoint = "/app/Data/Models/Epoch[8]_Loss[2.2832].pth"
    # )
    
   

if __name__ == "__main__":
    main()
