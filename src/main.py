import argparse
from typing import Final

import torch.optim as optim
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset.dataset import DatasetMode, MITBIHDataset
from dataset.datamodule import Mitbih_datamodule
import os

from model import Transformer_BPM_Regressor
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
    datamodule: Mitbih_datamodule, 
    model: nn.Module, 
    num_epochs: int,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_filename: str | None = None
   ) -> None:
    
    # Assicurati che la directory dei checkpoint esista
    os.makedirs(checkpoint_dir, exist_ok=True)
    #checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    start_epoch: int = 0
    
    if checkpoint_filename is not None:
        # Carica un checkpoint esistente se presente
        if os.path.exists(checkpoint_filename):
            print(f"Caricamento checkpoint da {checkpoint_filename}")
            checkpoint = torch.load(checkpoint_filename, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['loss']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Riprendi l'addestramento dall'epoca {start_epoch} con validation loss {best_val_loss:.4f}")


    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoca {epoch+1}/{num_epochs}")

        # --- Fase di Training ---
        model.train()
        train_loss = 0
        # Utilizza tqdm per visualizzare l'avanzamento del training
        train_loop = tqdm(train_dataloader, leave=False, desc=f"Training Epoca {epoch+1}")
        for batch_idx, (signal, bpm) in enumerate(train_loop):
            
            #print(signal.shape)
    
            signal = signal.to(device)
            bpm = bpm.to(device)

            optimizer.zero_grad()
            outputs = model(signal)
            loss = loss_function(outputs, bpm)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Aggiorna la descrizione di tqdm con la loss corrente
            train_loop.set_description(f"Training Epoca {epoch+1} Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)

        # --- Fase di Validation ---
        model.eval()
        val_loss = 0
        # Utilizza tqdm per visualizzare l'avanzamento della validation
        val_loop = tqdm(val_dataloader, leave=False, desc=f"Validation Epoca {epoch+1}")
        with torch.no_grad():
            for batch_idx, (signal, bpm) in enumerate(val_loop):
                
                signal = signal.to(device)
                bpm = bpm.to(device)
                
                outputs = model(signal)
                loss = loss_function(outputs, bpm)

                val_loss += loss.item()

                # Aggiorna la descrizione di tqdm con la loss corrente
                val_loop.set_description(f"Validation Epoca {epoch+1} Loss: {loss.item():.4f}")


        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoca {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        # --- Gestione Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_path = os.path.join(checkpoint_dir, f"Epoch_{epoch+1}-{avg_val_loss}")
            
            print(f"Validation loss migliorata. Salvataggio modello in {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_dir)

    print("Addestramento completato.")
    # # Opzionale: Carica il miglior modello addestrato prima di terminare la funzione
    # if os.path.exists(checkpoint_path):
    #      print(f"Caricamento del miglior modello da {checkpoint_path}")
    #      checkpoint = torch.load(checkpoint_path, map_location=device)
    #      model.load_state_dict(checkpoint['model_state_dict'])

    
    
    
def testModel(device: torch.device, datamodule: Mitbih_datamodule, model, **kwargs) -> None:
    pass

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
    
    # Imposta i parametri del dataset
    dataset_path = args.dataset_path
    MITBIHDataset.setDatasetPath(dataset_path)
    
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
        sample_rate=sample_rate, 
        window_size_t=args.window_size,
        window_stride_t=args.window_stride,
        num_workers=6,
        batch_size=12
    )
    
    #print(dataModule.train_dataset()[2])
    
    
    model = Transformer_BPM_Regressor(
        max_token=args.window_size*sample_rate,
        in_channels=2,
        d_model=args.d_model,
        head_num=8,
        num_encoder_layers=4,
        dim_feedforward=args.dff,
        dropout=args.dropout_rate,
    )
    
    print("Architettura del Modello:")
    print(model)
    
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando dispositivo: {device}")


    trainModel(
        device = device,
        datamodule = dataModule,
        model = model,
        num_epochs = 1,
        checkpoint_dir = setting.OUTPUT_PATH,
        checkpoint_filename = None
    )


if __name__ == "__main__":
    main()
    
    
#     # --- Esempio di Utilizzo ---
# if __name__ == "__main__":
#     # SOSTITUISCI con il percorso effettivo dove hai scaricato il dataset MIT-BIH
#     # ad esempio: './mit-bih-arrhythmia-database-1.0.0/'
    

#     # Crea un'istanza del dataset per il training
#     try:
       
#         print(f"Dataset di Training creato con {len(train_dataset)} campioni.")

#         # Puoi ora usare train_dataset con un DataLoader
#         from torch.utils.data import DataLoader
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#         # Esempio di iterazione sui primi batch
#         print("\nEsempio di batch dal Training DataLoader:")
#         for i, (segments, labels) in enumerate(train_loader):
#             print(f"Batch {i}:")
#             print(f"  Segmenti shape: {segments.shape}") # Dovrebbe essere [batch_size, 1, segment_length]
#             print(f"  Etichette shape: {labels.shape}")   # Dovrebbe essere [batch_size]
#             print(f"  Prime etichette nel batch: {labels[:10].tolist()}") # Stampa le prime 10 etichette
#             if i == 2: # Stampa solo i primi 3 batch per l'esempio
#                 break

#         print("-" * 30)

#         # Crea un'istanza del dataset per il testing
#         test_dataset = MITBIHDataset(data_path=MITBIH_PATH, mode=DatasetMode.TEST, segment_length=360)
#         print(f"\nDataset di Testing creato con {len(test_dataset)} campioni.")

#         test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Shuffle=False per testing/validation

#         # Esempio di recupero di un singolo campione
#         print("\nEsempio di recupero di un singolo campione dal dataset di Test:")
#         sample_segment, sample_label = test_dataset[0]
#         print(f"  Segmento shape: {sample_segment.shape}") # Dovrebbe essere [1, segment_length]
#         print(f"  Etichetta: {sample_label.item()}")

#     except FileNotFoundError as e:
#         print(f"Errore: {e}")
#         print("Assicurati di aver scaricato il dataset MIT-BIH e specificato il percorso corretto.")
#     except Exception as e:
#          print(f"Si è verificato un errore: {e}")