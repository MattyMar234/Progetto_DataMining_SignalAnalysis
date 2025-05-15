import argparse
from typing import Final

from dataset import DatasetMode, MITBIHDataset


#MITBIH_PATH = './mit-bih-arrhythmia-database-1.0.0/' # Assicurati che finisca con '/'
MITBIH_PATH: Final[str] = '/app/Dataset/mitbih_database'




def main():
    
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