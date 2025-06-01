import argparse
import math
from typing import Final
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from NN_component.resNet import ResNet1D
from dataset.dataset import DatasetDataMode, DatasetMode, MITBIHDataset, SampleType
from dataset.datamodule import Mitbih_datamodule
import os

from NN_component.weightedMSELoss import WeightedMSELoss
from NN_component.model import SimpleECGRegressor, Transformer_BPM_Regressor
from setting import *
import setting


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
            print(f"Caricamento checkpoint da {checkpoint}")
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            #best_val_loss = checkpoint_data['loss']
            start_epoch = checkpoint_data['epoch'] + 1
            print(f"Riprendi l'addestramento dall'epoca {start_epoch} con validation loss {best_val_loss:.4f}")


    
    

    with open(training_log_path, 'a') as log_file:
        if os.stat(training_log_path).st_size != 0:
            log_file.write(f"\n{'='*80}\n")

        log_file.write("Epoch, lr, Train_Loss, Train_MAE, Train_RMSE, Val_Loss, Val_MAE, Val_RMSE\n")

        
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoca {epoch+1}/{num_epochs}")

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
                
                #print(bpm, outputs)

                # print(f"bpm shape: {bpm.shape}")
                # print(f"outputs shape: {outputs.shape}")


                # print(outputs)
                # print(outputs.shape)
                
                # # print(bpm)
                # print(bpm.shape)


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
                    
                    #print(bpm)
                    
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


            print(f"Epoca {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

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

                print(f"Validation loss migliorata ({avg_val_loss:.4f}). Salvataggio modello in {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'loss': avg_val_loss, # Salva il miglior loss raggiunto
                }, checkpoint_path) # Salva nel file specificato





    print("Addestramento completato.")

def plot_confusion_matrix(cm, class_labels, epoch, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))
    plt.close()

def training_classification(
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
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    start_epoch: int = 0
    best_val_loss = float('inf')
    
    model.to(device)
    train_dataloader = dataModule.train_dataloader()
    val_dataloader = dataModule.val_dataloader()
    
    # Get class weights from the dataset
    class_weights = dataModule.get_train_dataset().getWeights().to(device)
    
    # Loss function: CrossEntropyLoss with weights and ignore_index=6
    # Class 6 (Unknown Beat Q) is ignored as per requirement
    loss_function = nn.CrossEntropyLoss(weight=class_weights, ignore_index=6)
    
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         
        factor=0.8,         
        patience=4,         
        threshold=0.0001,   
        threshold_mode='rel' 
    )
    
    if checkpoint is not None:
        if os.path.exists(checkpoint):
            print(f"Caricamento checkpoint da {checkpoint}")
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            start_epoch = checkpoint_data['epoch'] + 1
            print(f"Riprendi l'addestramento dall'epoca {start_epoch} con validation loss {best_val_loss:.4f}")

    class_labels = [
        "Normal", "SVEB", "VEB", "Fusion", 
        "Ventricular Flutter Wave", "Paced Beat", "Unknown Beat (Ignored)"
    ]

    with open(training_log_path, 'a') as log_file:
        if os.stat(training_log_path).st_size != 0:
            log_file.write(f"\n{'='*80}\n")

        log_file.write("Epoch, lr, Train_Loss, Val_Loss, Accuracy, F1_Macro, Precision_Macro, Recall_Macro\n")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoca {epoch+1}/{num_epochs}")

            # --- Fase di Training ---
            model.train()
            total_train_loss = 0
            
            train_preds = []
            train_targets = []

            train_loop = tqdm(train_dataloader, leave=False, desc=f"Training Epoca {epoch+1}")
            for batch_idx, (signal, labels) in enumerate(train_loop):
                signal = signal.to(device)
                labels = labels.squeeze(1).long().to(device) # Ensure labels are 1D and Long type
                #labels = F.one_hot(labels, num_classes=7).float() # Convert to one-hot with 7 classes
                
                # print(signal.shape)
                # print(labels.shape)
                
                optimizer.zero_grad()
                outputs = model(signal) # Outputs are logits
                
                #print(outputs.shape)
                
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                
                # Collect predictions and true labels, ignoring class 6
                preds = torch.argmax(outputs, dim=1)
                
                # Filter out ignored class 6 for metric calculation
                valid_indices = (labels != 6)
                train_preds.extend(preds[valid_indices].cpu().numpy())
                train_targets.extend(labels[valid_indices].cpu().numpy())
                
                train_loop.set_description(f"Training Epoca {epoch+1} Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_dataloader) # Divide by number of batches

            # --- Fase di Validation ---
            model.eval()
            total_val_loss = 0
            
            val_preds = []
            val_targets = []

            val_loop = tqdm(val_dataloader, leave=False, desc=f"Validation Epoca {epoch+1}")
            with torch.no_grad():
                for batch_idx, (signal, labels) in enumerate(val_loop):
                    signal = signal.to(device)
                    labels = labels.squeeze(1).long().to(device) # Ensure labels are 1D and Long type
                    #labels = F.one_hot(labels, num_classes=7).float() # Convert to one-hot with 7 classes
                    
                    outputs = model(signal)
                    loss = loss_function(outputs, labels)
                    
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    
                    # Filter out ignored class 6 for metric calculation
                    valid_indices = (labels != 6)
                    val_preds.extend(preds[valid_indices].cpu().numpy())
                    val_targets.extend(labels[valid_indices].cpu().numpy())
                    
                    val_loop.set_description(f"Validation Epoca {epoch+1} Loss: {loss.item():.4f}")

            avg_val_loss = total_val_loss / len(val_dataloader) # Divide by number of batches

            # Calculate metrics
            # Ensure that the labels passed to sklearn metrics do not contain the ignored class
            # And that the predictions are also filtered accordingly.
            # The unique labels for confusion matrix should exclude 6.
            unique_labels_for_metrics = [i for i in range(len(class_labels)) if i != 6]

            cm = confusion_matrix(val_targets, val_preds, labels=unique_labels_for_metrics)
            plot_confusion_matrix(cm, [class_labels[i] for i in unique_labels_for_metrics], epoch + 1, confusion_matrix_dir)

            accuracy = accuracy_score(val_targets, val_preds)
            f1_macro = f1_score(val_targets, val_preds, average='macro', labels=unique_labels_for_metrics)
            precision_macro = precision_score(val_targets, val_preds, average='macro', labels=unique_labels_for_metrics)
            recall_macro = f1_score(val_targets, val_preds, average='macro', labels=unique_labels_for_metrics) # F1-score is harmonic mean of precision and recall

            print(f"Epoca {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}, Precision-Macro: {precision_macro:.4f}, Recall-Macro: {recall_macro:.4f}")

            scheduler.step(avg_val_loss)

            log_file.write(f"{epoch+1}, {optimizer.param_groups[0]['lr']:.6f}, {avg_train_loss:.6f}, {avg_val_loss:.6f}, {accuracy:.6f}, {f1_macro:.6f}, {precision_macro:.6f}, {recall_macro:.6f}\n")
            log_file.flush()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(checkpoint_dir, f"Epoch[{epoch+1}]_Loss[{avg_val_loss:.4f}].pth")
                print(f"Validation loss migliorata ({avg_val_loss:.4f}). Salvataggio modello in {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'loss': avg_val_loss,
                }, checkpoint_path)

    print("Addestramento completato.")

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

   
    print(f"Caricamento checkpoint da {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
   
    test_dataloader = datamodule.test_dataloader()
    batch_size = test_dataloader.batch_size
    # Disabilita il calcolo dei gradienti durante la valutazione
    
    print("Avvio valutazione sul set di test...")
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
                print(f"Target: {bpm[i]} Predected: {outputs[i]} Loss: -")

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

    print("\n--- Risultati Test Set ---")
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Test MAE: {avg_test_mae:.6f}")
    print(f"Test RMSE: {avg_test_rmse:.6f}")
    print("--------------------------")

    metrics = {
        'test_loss': avg_test_loss,
        'test_mae': avg_test_mae,
        'test_rmse': avg_test_rmse
    }

    return metrics
  
    


def main():

    parser = argparse.ArgumentParser(description="Script per configurare e addestrare un modello Transformer sui dati MIT-BIH.")
    
    # Parametri per la struttura del modello Transformer
   

    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help=f"Numero di strati del Transformer (default: {NUM_LAYERS})")
    parser.add_argument("--d_model", type=int, default=D_MODEL, help=f"Dimensione del modello (default: {D_MODEL})")
    parser.add_argument("--num_heads", type=int, default=NUM_HEADS, help=f"Numero di teste di attenzione (default: {NUM_HEADS})")
    parser.add_argument("--dff", type=int, default=DFF, help=f"Dimensione del feed-forward (default: {DFF})")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE, help=f"Tasso di dropout (default: {DROPOUT_RATE})")
    
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

    
    # MITBIHDataset.setDatasetPath(MITBIH_PATH)
    
    # train_dataset = MITBIHDataset(
    #     mode=DatasetMode.TRAINING, 
    #     sample_rate=360,
    #     sample_per_window=360*10,  # 10 secondi
    #     sample_per_side=360*5,  # 5 secondi
    # )
    
    dataModule = Mitbih_datamodule(
        args.dataset_path, 
        datasetDataMode=DatasetDataMode.BEAT_CLASSIFICATION,
        sample_rate=sample_rate, 
        sample_per_window=sample_rate,#args.window_size,
        num_workers=4,
        batch_size=12
    )
    
    # dataset = dataModule.get_train_dataset()
    # path = os.path.join(setting.DATA_FOLDER_PATH,'BEATS', f'plot.png')
    # dataset.plot_class_distribution(path, False)
    # dataset.plot_record_with_annotations('101','plot2.png', 60 )
    
    #dataModule.print_all_window(columNumber=0, save_path=path, dataset=dataset)
    # for i in range(len(dataset)):
    #     path = os.path.join(setting.DATA_FOLDER_PATH,'BEATS', f'test{i}_plot.png')
    #     dataModule.print_window(path, dataset, i)
    
    
    
    
    #dataModule.print_all_training_ecg_signals(os.path.join(setting.DATA_FOLDER_PATH, 'training_plots'))
    #dataModule.print_training_plot_bpm_distribution(os.path.join(setting.DATA_FOLDER_PATH, 'training_plots'))
    #dataModule.print_validation_plot_bpm_distribution(os.path.join(setting.DATA_FOLDER_PATH, 'validation_plots'))
    
    #dataModule.print_training_record('207',os.path.join(setting.DATA_FOLDER_PATH, 'test_plot'))
    
    
   
    model = ResNet1D(in_channels_signal=2, output_dim=7)

    
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
    
    print("Architettura del Modello:")
    print(model)
    
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando dispositivo: {device}")

    training_classification(
        device = device,
        dataModule = dataModule,
        model = model,
        num_epochs = 40,
        training_log_path=os.path.join(setting.LOGS_FOLDER, 'training_logs.txt'),
        checkpoint_dir = setting.OUTPUT_PATH,
        confusion_matrix_dir=setting.LOGS_FOLDER
        #checkpoint = "/app/Data/Models/Epoch[1]_Loss[inf].pth"
    
    )

    # trainModel(
    #     device = device,
    #     dataModule = dataModule,
    #     model = model,
    #     num_epochs = 40,
    #     training_log_path=os.path.join(setting.LOGS_FOLDER, 'training_logs.txt'),
    #     checkpoint_dir = setting.OUTPUT_PATH,
    #     #checkpoint = "/app/Data/Models/Epoch[1]_Loss[inf].pth"
    # )
    
    # test_model(
    #     device = device,
    #     datamodule = dataModule,
    #     model = model,
    #     checkpoint = "/app/Data/Models/Epoch[8]_Loss[2.2832].pth"
    # )
    
   

if __name__ == "__main__":
    main()
