import argparse
from typing import Final

from dataset import DatasetMode, MITBIHDataset
import os

from setting import *


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

    
    MITBIHDataset.setDatasetPath(MITBIH_PATH)
    
    train_dataset = MITBIHDataset(
        mode=DatasetMode.TRAINING, 
        sample_rate=360,
        sample_per_window=360*10,  # 10 secondi
        sample_per_side=360*5,  # 5 secondi
    )
    
    print(f"Dataset di Training creato con {len(train_dataset)} campioni.")
    
    train_dataset.plot_all_windows_for_record('207')


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