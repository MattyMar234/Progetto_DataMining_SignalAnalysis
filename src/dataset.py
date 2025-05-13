import torch
import wfdb
import numpy as np
import os
from torch.utils.data import Dataset
from enum import Enum, auto
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


# Mappatura dei tipi di battito alle classi AAMI
# Raggruppamento AAMI Recommended Practice (ANSI/AAMI EC38:2000)
# N: Normal beat (.'N', 'L', 'R', 'B') -> 0
# SVEB: Supraventricular ectopic beat ('A', 'a', 'J', 'S') -> 1
# VEB: Ventricular ectopic beat ('V', 'E') -> 2
# F: Fusion beat ('F') -> 3
# Q: Unknown beat ('Q', '/') -> 4
# Ignoriamo altri simboli di annotazione non relativi ai battiti.
AAMI_CLASSES = {
    'N': 0, 'L': 0, 'R': 0, 'B': 0, '|': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1, 'j': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, '~': 4,
    '+': 5, '\"': 5,
    'X': 6, 'x' : 6,

}


# Definizione delle modalità del dataset
class DatasetMode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    
class SampleType(Enum):
    NORMAL = auto()
    SVEB = auto()
    VEB = auto()
    FUSION = auto()
    UNKNOWN = auto()
    NOSIE = auto() # Per campioni che non possono essere classificati
    START_SEQUENZE = auto() # Per campioni che non possono essere classificati
    
    @classmethod
    def to_Label(cls, type: str) -> 'SampleType':
        
        idx = AAMI_CLASSES.get(type, -1)
        
        match idx:
            case 0:
                return SampleType.NORMAL
            case 1:
                return SampleType.SVEB
            case 2:
                return SampleType.VEB
            case 3:
                return SampleType.FUSION
            case 4:
                return SampleType.UNKNOWN
            case 5:
                return SampleType.START_SEQUENZE
            case 6:
                return SampleType.NOSIE
            case _:
                raise ValueError(f"Tipo di battito sconosciuto: {type}. Non può essere convertito in SampleType.")


# Tipi di battito da includere (quelli mappati nelle classi AAMI)
VALID_BEAT_SYMBOLS = list(AAMI_CLASSES.keys())
  
class MITBIHDataset(Dataset):
    """
    Classe PyTorch Dataset adattata per il database MIT-BIH Arrhythmia
    con dati in formato CSV per segnali e TXT per annotazioni.

    Carica segmenti di ECG centrati attorno ai battiti e le loro annotazioni
    corrispondenti per le modalità di training, validation o testing.
    """
    _MLII_COL: str = 'MLII'
    _V1_COL: str = 'V1'
    _V2_COL: str = 'V2'
    _V3_COL: str = 'V3'
    _V4_COL: str = 'V4'
    _V5_COL: str = 'V5'
    
    
    _FILE_CHEKED: bool = False
    _DATASET_PATH: None | str = None
    
    _RECORDS_MLII_V5_V2 = ['104', '102']
    _RECORDS_MLII_V1 = ['101','105','106', '107', '108', '109','111', '112','113','115', '116','118','119','121','122',
                        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212','213', '214','215','217',
                        '219', '220', '221', '222', '223', '228', '230','231','232','233','234']
    _RECORDS_MLII_V2 = ['117', '103']
    _RECORDS_MLII_V4 = ['124']
    _RECORDS_MLII_V5 = ['100','114','123']
    
    _ALL_RECORDS: list = _RECORDS_MLII_V1
            
    

    _RISOLUZIONE_ADC: int = 11
    _MIN_VALUE: int = 0
    _MAX_VALUE: int = 2**_RISOLUZIONE_ADC - 1
    _MAX_SAMPLE_NUM: int = 650000
    
    
    def __new__(cls, *args, **kwargs):
        return super(MITBIHDataset, cls).__new__(cls)

    @classmethod
    def setDatasetPath(cls, path: str):
        """
        Imposta il percorso del dataset.
        """
        
        if cls._FILE_CHEKED and cls._DATASET_PATH == path:
            print(f"Il percorso del dataset è già impostato su: {cls._DATASET_PATH}")
            return
        
        cls._DATASET_PATH = path
        cls._FILE_CHEKED = False
        
        if not os.path.isdir(path):
            raise FileNotFoundError(f"La directory specificata non esiste: {path}")
        print(f"Percorso del dataset impostato su: {cls._DATASET_PATH}")
        
 

        # Controlla se i file CSV e TXT esistono
        for record_name in cls._ALL_RECORDS:
            csv_filepath = os.path.join(path, f"{record_name}.csv")
            txt_filepath = os.path.join(path, f"{record_name}annotations.txt")

            if not os.path.exists(csv_filepath):
                raise FileNotFoundError(f"File CSV non trovato: {csv_filepath}")
            if not os.path.exists(txt_filepath):
                raise FileNotFoundError(f"File TXT non trovato: {txt_filepath}")

        cls._FILE_CHEKED = True
        
    
    def __init__(self, *, mode: DatasetMode, sample_rate: int = 360, sample_per_window: int = 360, sample_per_side: int = 180):
        
        if not MITBIHDataset._DATASET_PATH:
            raise ValueError("Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo.")
        

        self.mode = mode
        self.sample_rate = sample_rate
        self.samples_per_window = sample_per_window
        self.samples_per_side = sample_per_side
         # Normalizza il nome della colonna

        # Definizione dello split dei record per training, validation e test
        # Questo è uno split comune utilizzato nella letteratura.
        # Usa gli stessi nomi base dei file CSV/TXT (es. '100')
        train_records = [
            '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
            '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
            '223', '230'
        ]
        validation_records = [
            '100', '103', '105', '107', '111', '113', '117', '123', '200', '202',
            '210', '212', '213', '214'
        ]
        test_records = [
            '217', '219', '221', '222', '228', '231', '232', '233', '234', '104',
            '121'
        ]

        if mode == DatasetMode.TRAINING:
            self.record_list = train_records
            
        elif mode == DatasetMode.VALIDATION:
            self.record_list = validation_records
            
        elif mode == DatasetMode.TEST:
            self.record_list = test_records
            
        else:
            raise ValueError(f"Modalità non valida: {mode}")

        print(f"Caricamento dati per la modalità {self.mode.name} dai record: {self.record_list}")
        
        
    
        self._signals_dict = {} # Dizionario per memorizzare i segnali
        self._windows: Dict[int, Dict[str, any]] = {} 
        self._load_data()
        
    def plot_windows(self):
        # --- Plot del segnale ---
        output_dir = os.path.join(self.data_path, "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"{record_name}_ecg_plot.png")
        
        plt.figure(figsize=(12, 4))
        plt.plot(signal[:2000], label='Segnale ECG')
        plt.title(f"Segnale ECG - Record {record_name}")
        plt.xlabel("Campioni")
        plt.ylabel("Ampiezza")
        plt.legend()
        plt.grid()
        
        # Aggiungi linee verticali ai punti specificati
        vertical_lines = [7, 83, 396, 711, 1032, 1368, 1712]
        for x in vertical_lines:
            plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
        
        
        plt.savefig(output_filepath)
        plt.close()

    def _load_data(self):
        
        windows_counter:int = 0

        """Carica i dati dai record selezionati (CSV per segnale, TXT per annotazioni)."""
        for record_name in self.record_list:
            csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
            txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")

       
            # --- Leggi il segnale dal file CSV ---
            # Si aspetta colonne: 'sample #','MLII','V5'
            df = pd.read_csv(csv_filepath)
            
            for idx, col in enumerate(df.columns):
                df = df.rename(columns={df.columns[idx]: df.columns[idx].replace('\'', '')}) # Rinomina la prima colonna in 'sample #'

            signal: torch.Tensor | None = None
            
            if MITBIHDataset._MLII_COL in df.columns:
                signal = torch.from_numpy(df[MITBIHDataset._MLII_COL].values).unsqueeze(0)
            
            if MITBIHDataset._V1_COL in df.columns:
                if signal is None:
                    signal = torch.from_numpy(df[MITBIHDataset._V1_COL].values).unsqueeze(0)
                else:
                    signal = torch.cat((signal, torch.from_numpy(df[MITBIHDataset._V1_COL].values).unsqueeze(0)), dim=0)
            
            if MITBIHDataset._V2_COL in df.columns:
                if signal is None:
                    signal = torch.from_numpy(df[MITBIHDataset._V2_COL].values).unsqueeze(0)
                else:
                    signal = torch.cat((signal, torch.from_numpy(df[MITBIHDataset._V2_COL].values).unsqueeze(0)), dim=0)
           
            # if MITBIHDataset._V3_COL in df.columns:
            #     if signal is None:
            #         signal = torch.from_numpy(df[MITBIHDataset._V3_COL].values).unsqueeze(0)
            #     else:
            #         signal = torch.cat((signal, torch.from_numpy(df[MITBIHDataset._V3_COL].values).unsqueeze(0)), dim=0)
            
            if MITBIHDataset._V4_COL in df.columns:
                if signal is None:
                    signal = torch.from_numpy(df[MITBIHDataset._V4_COL].values).unsqueeze(0)
                else:
                    signal = torch.cat((signal, torch.from_numpy(df[MITBIHDataset._V4_COL].values).unsqueeze(0)), dim=0)
            
            if MITBIHDataset._V5_COL in df.columns:
                if signal is None:
                    signal = torch.from_numpy(df[MITBIHDataset._V5_COL].values).unsqueeze(0)
                else:
                    signal = torch.cat((signal, torch.from_numpy(df[MITBIHDataset._V5_COL].values).unsqueeze(0)), dim=0)
            
           

            # Normalizza il segnale
            signal = signal.float() 
            signal = (signal - MITBIHDataset._MIN_VALUE) / (MITBIHDataset._MAX_VALUE - MITBIHDataset._MIN_VALUE) 
            
          
            
            # Salva il segnale per il record corrente
            self._signals_dict[record_name] = signal 
            
            
            windows_number = int((len(signal[0]) - self.samples_per_window) / self.samples_per_side) + 1
            record_windows: list = []
            
           
            #creazione delle finestre
            for i in range(windows_number):
                start = i * self.samples_per_side
                end = start + self.samples_per_window
                
                # Crea una finestra per il segnale
                window = {
                    'start': start,
                    'end': end,
                    'signal': signal[:, start:end],
                    'record_name': record_name,
                    'beat_number': 0,
                    'BPM': 0,
                    'beat_positions': [],
                    'beat_labels': [],   
                    'beat_time': [],
                }
                
                record_windows.append(window)
            
            #print(f"Record {record_name} ha {len(record_windows)} finestre.")
            
            window_pointer: int = 0
            current_window = record_windows[window_pointer]
            window_start = current_window['start']
            window_end = current_window['end']
            
            with open(txt_filepath, 'r') as f:
                f.readline() # Salta la prima riga (header)
                
                for line in f:

                    # Rimuovi spazi bianchi
                    line = line.strip()
                    
                    # Salta linee vuote o commenti
                    if not line or line.startswith('#'): 
                        continue

                    # Dividi la riga in base agli spazi e ignore le stringe vuote
                    parts = line.split() 

                    # Una riga di annotazione valida dovrebbe avere almeno 3 parti (Time, Sample #, Type)
                    if len(parts) < 3:
                        raise ValueError(f"Riga di annotazione non valida: {line}.")

                    beat_type = SampleType.to_Label(parts[2])
                    sample = int(parts[1]) # Indice del campione
                    time = self._formatTime(parts[0]) # Tempo in secondi
                    
                    #cerca la finestra in cui si trova il campione
                    while not (window_start <= sample <= window_end):
                        window_pointer += 1
                        current_window = record_windows[window_pointer]
                        window_start = current_window['start']
                        window_end = current_window['end']
                    
                                        
                    match beat_type:
                        case SampleType.NORMAL | SampleType.SVEB | SampleType.VEB | SampleType.FUSION:
                            current_window['beat_positions'].append(sample)
                            current_window['beat_labels'].append(beat_type)
                            current_window['beat_number'] += 1
                            current_window['beat_time'].append(time)
                            
                        
                        case SampleType.UNKNOWN | SampleType.START_SEQUENZE | SampleType.NOSIE:
                            pass
                        
                        case _:
                            # Se il tipo di battito non è valido, ignora
                            print(f"Tipo di battito sconosciuto: {parts[2]}. Ignorato.")
                            continue
                        
                        # #Controlla la colonna 'Aux' se esiste e contiene (simbolo)
                        # if len(parts) > 6 and parts[6].startswith('(') and len(parts[6]) > 1:
                        #     beat_symbol_raw = parts[6].strip('()')
                        #     if beat_symbol_raw in VALID_BEAT_SYMBOLS:
                        #         aux = beat_symbol_raw
                    
      
            for i in range(len(record_windows)):
                current_window = record_windows[i]
                current_window['BPM'] = (current_window['beat_number'] / (current_window['end'] - current_window['start'])) * 60
                #print(f"Finestra {i} BPM: {current_window['BPM']}")
                
                self._windows[windows_counter] = current_window
                windows_counter += 1
           
    def _formatTime(self, time: str) -> float:        
        parts = time.split(':')
        split = parts[-1].split('.')
        milliseconds = int(split[1])
        seconds = int(split[0])           
        minutes = int(parts[-2])
        if len(parts) > 2:
            hours = parts[-3]
        else:
            hours = 0
        
        return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)
 
        

    def __len__(self) -> int:
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self._windows.keys())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recupera un singolo campione e la sua etichetta.

        Args:
            idx (int): Indice del campione.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Una tupla contenente il tensore
                                              del segmento ECG e il tensore
                                              dell'etichetta (classe AAMI).
                                              Il segmento ECG avrà la forma (1, segment_length)
                                              per essere compatibile con layer convoluzionali
                                              che si aspettano (canali, lunghezza).
        """
        # segment, label = self.data[idx]

        # # Converti in tensori PyTorch
        # segment_tensor = torch.tensor(segment, dtype=torch.float32)
        # label_tensor = torch.tensor(label, dtype=torch.long) # Le etichette sono tipicamente LongTensor per la classificazione

        # # Aggiungi una dimensione per i canali (necessario per molti modelli come le CNN)
        # # (segment_length,) -> (1, segment_length)
        # segment_tensor = segment_tensor.unsqueeze(0)

        # return segment_tensor, label_tensor
        
        return self._windows[idx]


