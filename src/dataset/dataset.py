import random
from sklearn.model_selection import train_test_split
import torch
import wfdb
import numpy as np
import os
from torch.utils.data import Dataset
from enum import Enum, auto
from typing import Any, Dict, Final, List, Tuple
import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


from sklearn.utils.class_weight import compute_class_weight

# Definizione delle modalità del dataset
class DatasetMode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    
class DatasetDataMode(Enum):
    BEAT_CLASSIFICATION = auto()
    BPM_REGRESSION = auto()
    
class SampleType(Enum):
    
    #BEAT
    NORMAL_N = "N"
    NORMAL_L = "L"
    NORMAL_R = "R"
    NORMAL_B = "B"
    NORMAL_LINE = "|"
    
    SVEB_A = "A"
    SVEB_a = "a"
    SVEB_J = "J"
    SVEB_j = "j"
    SVEB_S = "S"
    
    VEB_V = "V"
    VEB_E = "E"
    VEB_e = "e"
    
    FUSION_F = "F"
    FUSION_f = "f"
   
    VENTRICULAR_FLUTTER_WAVE = "!"
    UNKNOWN_BEAT_Q = "Q"
    PACED_BEAT = "/"
  
    #ANNOTATION
    CHANGE_IN_SIGNAL_QUALITY = "~"
    NOISE = "X x"
    START_NOISE = "["
    END_NOISE = "]"
    START_SEG_PLUS = "+"
    COMMENT = "\""
    
    # ---- BEAT TO NUMBER MAP ---- #
    __beat_to_number_map = {

        NORMAL_N: 0, NORMAL_L: 0, NORMAL_R: 0, NORMAL_B: 0, NORMAL_LINE: 0,
        SVEB_A: 1, SVEB_a: 1, SVEB_J: 1, SVEB_j: 1, SVEB_S: 1,
        VEB_V: 2, VEB_E: 2, VEB_e: 2,
        FUSION_F: 3, FUSION_f: 3,
        VENTRICULAR_FLUTTER_WAVE: 4, 
        PACED_BEAT: 5,
        UNKNOWN_BEAT_Q: 6, 
    }
    
    @classmethod
    def to_Label(cls, type: str) -> 'SampleType':
        if not isinstance(type, str) or len(type) == 0:
            raise ValueError(f"Tipo di battito non valido: {type}")

        # Mappa il carattere a un oggetto SampleType usando i valori degli enum
        type_char = type.strip()[0]
        for member in cls:
            if type_char in member.value.split():
                return member
        raise ValueError(f"Tipo di battito sconosciuto: {type_char}. Non può essere convertito in SampleType.")
    
    @classmethod  
    def isBeat(cls, annotation: 'SampleType') -> bool:
        return annotation in {
            cls.NORMAL_N, cls.NORMAL_L, cls.NORMAL_R, cls.NORMAL_B, cls.NORMAL_LINE,
            cls.SVEB_A, cls.SVEB_a, cls.SVEB_J, cls.SVEB_j, cls.SVEB_S,
            cls.VEB_V, cls.VEB_E, cls.VEB_e,
            cls.FUSION_F, cls.FUSION_f,
            cls.VENTRICULAR_FLUTTER_WAVE, cls.UNKNOWN_BEAT_Q, cls.PACED_BEAT
        }
    
    @classmethod
    def isTag(cls, annotation: 'SampleType') -> bool:
        return annotation in {
            cls.CHANGE_IN_SIGNAL_QUALITY, cls.NOISE, cls.START_NOISE,
            cls.END_NOISE, cls.START_SEG_PLUS, cls.COMMENT
        }
        
    @classmethod
    def mapBeatToNumber(cls, beat: 'SampleType') -> int:
        if beat not in cls.__beat_to_number_map:
            raise ValueError(f"Il beat {beat} non ha una mappatura numerica definita.")
        return cls.__beat_to_number_map[beat]
    
  
class MITBIHDataset(Dataset):
    """
    Classe PyTorch Dataset adattata per il database MIT-BIH Arrhythmia
    con dati in formato CSV per segnali e TXT per annotazioni.
    """
    _MLII_COL: str = 'MLII'
    _V1_COL: str = 'V1'
    _V2_COL: str = 'V2'
    _V3_COL: str = 'V3'
    _V4_COL: str = 'V4'
    _V5_COL: str = 'V5'
    
    
    _FILES_CHEKED: bool = False
    _DATASET_PATH: None | str = None
    
    _RECORDS_V5_V2   = ['104', '102']
    _RECORDS_MLII_V1 = ['101','105','106', '107', '108', '109','111', '112','113','115', '116','118','119','121','122',
                        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212','213', '214','215','217',
                        '219', '220', '221', '222', '223', '228', '230','231','232','233','234']
    _RECORDS_MLII_V2 = ['117', '103']
    _RECORDS_MLII_V4 = ['124']
    _RECORDS_MLII_V5 = ['100','114','123']
    
    ALL_RECORDS: Final[list] = _RECORDS_MLII_V1
    TRAINING_RECORDS: list = []
    VALIDATION_RECORDS: list = []
    TEST_RECORDS: list = []
    

    _RISOLUZIONE_ADC: int = 11
    _MIN_VALUE: int = 0
    _MAX_VALUE: int = 2**_RISOLUZIONE_ADC - 1
    _MAX_SAMPLE_NUM: int = 650000
    _MAX_BPM=260
    _MIN_BPM=0
    
    
    def __new__(cls, *args, **kwargs):
        return super(MITBIHDataset, cls).__new__(cls)


    @classmethod
    def resetDataset(cls) -> None:
        pass

    @classmethod
    def initDataset(cls, path: str):
        """
        Imposta il percorso del dataset e carica staticamente tutti i dati.
        Questo metodo dovrebbe essere chiamato una volta all'inizio del programma.
        """
        if cls._FILES_CHEKED and cls._DATASET_PATH == path:
            print(f"Il percorso del dataset è già impostato su: {cls._DATASET_PATH}")
            return
        
        cls._DATASET_PATH = path
        cls._FILES_CHEKED = False
        
        if not os.path.isdir(path):
            raise FileNotFoundError(f"La directory specificata non esiste: {path}")
        print(f"Percorso del dataset impostato su: {cls._DATASET_PATH}")
        
        min_list = []
        max_list = []
 
        # Controlla se i file CSV e TXT esistono
        for record_name in cls.ALL_RECORDS:
            csv_filepath = os.path.join(path, f"{record_name}.csv")
            txt_filepath = os.path.join(path, f"{record_name}annotations.txt")

            if not os.path.exists(csv_filepath):
                raise FileNotFoundError(f"File CSV non trovato: {csv_filepath}")
            if not os.path.exists(txt_filepath):
                raise FileNotFoundError(f"File TXT non trovato: {txt_filepath}")

   
            current_record_name = record_name
            csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
            #txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")

    
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
            max_value = torch.max(signal).item()
            min_value = torch.min(signal).item()

            max_list.append(max_value)
            min_list.append(min_value)

        cls._MAX_VALUE = max(max_list)
        cls._MIN_VALUE = min(min_list)
        print(f"Valore massimo trovato: {cls._MAX_VALUE}")
        print(f"Valore minimo trovato: {cls._MIN_VALUE}")
        
        cls._FILES_CHEKED = True
        
        
    @classmethod
    def init_dataset(cls) -> None:
        assert cls._FILES_CHEKED, "files del dataset non verificati"
        
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        train_indices, temp_indices = train_test_split(cls.ALL_RECORDS, test_size=(val_ratio + test_ratio), random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        cls.TRAINING_RECORDS = train_indices
        cls.VALIDATION_RECORDS = val_indices
        cls.TEST_RECORDS = test_indices
        
        
    
    def __init__(self, *, mode: DatasetMode, dataMode: DatasetDataMode, sample_rate: int = 360, sample_per_window: int = 360, sample_per_stride: int = 180):
        assert MITBIHDataset._DATASET_PATH, "Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo."
        assert MITBIHDataset._FILES_CHEKED, "files del dataset non verificati"
        
   
        self._mode: DatasetMode = mode
        self._dataMode: DatasetDataMode = dataMode
        
        self._sample_rate: int = sample_rate
        self._samples_per_window: int = sample_per_window
        self._samples_per_side:int = sample_per_stride
        
        # lista dei record da utilizzare
        self._record_list: list | None = None               
        
        # Dizionario per memorizzare il segnale di ogni record
        self._signals_dict: Dict[str, torch.Tensor] = {}
        
        # Dizionario per memorizzare le annotazioni di ogni segnale
        self._signals_annotations_dict: Dict[str, List[Dict[str, any]]] = {}
        
        #dizionario delle finestre di dati     
        self._windows: Dict[int, Dict[str, any]] = {}       
        self._record_windows: Dict[str, Dict[int, Dict[str, any]]] = {}
        self._BPM_toWindows: Dict[int, list] = {}
        
        
        match self._mode:
            case DatasetMode.TRAINING:
                self._record_list = MITBIHDataset.TRAINING_RECORDS
       
            case DatasetMode.VALIDATION:
                self._record_list = MITBIHDataset.VALIDATION_RECORDS
            
            case DatasetMode.TEST:
                self._record_list = MITBIHDataset.TEST_RECORDS
            
            case _ :
                raise ValueError(f"Modalità non valida: {self._mode}")
        
        print(f"Caricamento dati per la modalità {self._mode.name} dai record: {self._record_list}")
        
        self._load_signals()
        
        match self._dataMode:
            case DatasetDataMode.BPM_REGRESSION: 
                self._load_data_for_BPM_regression()
                
            case DatasetDataMode.BEAT_CLASSIFICATION: 
                self._load_data_for_Beat_classification()
                
            case _ :
                raise ValueError(f"Modalità per i dati non valida: {self._dataMode}")
        
        

    def _load_signals(self, fill_and_concat_missing_channels: bool = False) -> None:
        
        colums_list: list = [
            MITBIHDataset._MLII_COL,
            MITBIHDataset._V1_COL,
            MITBIHDataset._V2_COL,
            #MITBIHDataset._V3_COL, non presente nel dataset
            MITBIHDataset._V4_COL,
            MITBIHDataset._V5_COL
        ]
        
        for record_name in self._record_list:
            csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
            txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")
            signal: torch.Tensor | None = None
            
            #========================================================================#
            # ESTRAGGO IL SEGNALE
            #========================================================================#
            # --- Leggi il segnale dal file CSV ---
            df = pd.read_csv(csv_filepath)
            
            for idx, col in enumerate(df.columns):
                df = df.rename(columns={df.columns[idx]: df.columns[idx].replace('\'', '')})

            for col in colums_list:
                if col in df.columns:
                    data = torch.from_numpy(df[col].values).unsqueeze(0)
                elif fill_and_concat_missing_channels:
                    data = torch.zeros(MITBIHDataset._MAX_SAMPLE_NUM).unsqueeze(0)
            
                if signal is None:
                    signal = data
                else:
                    signal = torch.cat(data, dim=0)
                
            # Normalizza il segnale
            signal = signal.float() 
            signal = (signal - MITBIHDataset._MIN_VALUE) / (MITBIHDataset._MAX_VALUE - MITBIHDataset._MIN_VALUE) 
            
            # Salva il segnale per il record corrente
            self._signals_dict[record_name] = signal 
            
            #========================================================================#
            # ESTRAGGO LE ANNOTAZIONI
            #========================================================================#
            with open(txt_filepath, 'r') as f:
                f.readline() # Salta la prima riga (header)
                
                dataDict_List: List[Dict[str, any]] = []
                
                for line in f:
                    line = line.strip()  # Rimuovi spazi bianchi
                    parts = line.split() # Dividi la riga in base agli spazi e ignore le stringe vuote

                    # Una riga di annotazione valida dovrebbe avere almeno 3 parti (Time, Sample #, Type)
                    if len(parts) < 3:
                        raise ValueError(f"Riga di annotazione non valida: {line}.")

                    dataDict_List.append({
                        "annotation" : SampleType.to_Label(parts[2]),   #tipologia di sample
                        "sample_pos" : int(parts[1]),                   # Indice del campione
                        "time" : self._formatTime(parts[0])             # Tempo in secondi
                    })
                    
            # Salva le annotazioni per il record corrente
            self._signals_annotations_dict[record_name] = dataDict_List 

    def _load_data_for_BPM_regression(self):
        """Carica i dati dai record selezionati (CSV per segnale, TXT per annotazioni)."""
        
        windows_counter:int = 0
        current_record_name:str | None = None
        BPM_value = {n: 0 for n in range(0,261)}
        
        try:
    
            for record_name in self._record_list:
                current_record_name = record_name
                signal = self._signals_dict[record_name]
                annotations = self._signals_annotations_dict[record_name]
                

                #========================================================================#
                # REALIZZAZIONE DELLE FINESTRE
                #========================================================================#
                windows_number:float = ((len(signal[0]) - self._samples_per_window) / self._samples_per_side) + 1
                windows_number_int = int(windows_number)
                record_windows: list = []
            
                #creazione delle finestre
                for i in range(windows_number_int):
                    start = i * self._samples_per_side
                    end = start + self._samples_per_window
                    
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
                        'tag' : [],
                        'tag_positions' : [],  
                        'beat_time': [],
                    }
                    
                    record_windows.append(window)
                
                # Se ci sono campioni rimanenti, crea una finestra finale
                if windows_number - windows_number_int > 0:
                    start = MITBIHDataset._MAX_SAMPLE_NUM - 1 - self._samples_per_window
                    end = start + self._samples_per_window
                    
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
                        'tag' : [],
                        'tag_positions' : [],  
                        'beat_time': [],
                    }
                    
                    record_windows.append(window)
                
                #print(f"Record {record_name} ha {len(record_windows)} finestre.")
                
                #========================================================================#
                # ASSEGNAZIONE DEI BEAT E TAG ALLE FINESTRE
                #========================================================================#
                window_pointer: int = 0
                current_window = record_windows[window_pointer]
                window_start = current_window['start']
                window_end = current_window['end']
                
               
                for ann in annotations:
                    beat_type = ann["annotation"]
                    sample_pos = ann["sample_pos"]
                    time = ann["time"]
                    
                    #Verifico il tipo di annotazione
                    match beat_type:
                        
                        #Se è un beat
                        case SampleType.NORMAL_N | SampleType.NORMAL_L | SampleType.NORMAL_R | SampleType.NORMAL_B | SampleType.NORMAL_LINE\
                            | SampleType.SVEB_A | SampleType.SVEB_a | SampleType.SVEB_J | SampleType.SVEB_j | SampleType.SVEB_S\
                            | SampleType.VEB_V | SampleType.VEB_E | SampleType.VEB_e \
                            | SampleType.FUSION_F | SampleType.FUSION_f \
                            | SampleType.UNKNOWN_BEAT_Q | SampleType.PACED_BEAT | SampleType.VENTRICULAR_FLUTTER_WAVE:
                                
                                        
                            #for w in range(max(0 , i_pointer), min(i_pointer+2, len(record_windows))):
                            for w in range(0, len(record_windows)):
                                current_window = record_windows[w]
                                window_start = current_window['start']
                                window_end = current_window['end']
                                
                                if window_start <= sample_pos <= window_end:
                                    current_window['beat_positions'].append(sample_pos)
                                    current_window['beat_labels'].append(beat_type)
                                    current_window['beat_number'] += 1
                                    current_window['beat_time'].append(time)
                            
                        #se è un tag utile      
                        case SampleType.START_SEG_PLUS | SampleType.COMMENT | SampleType.END_NOISE | SampleType.START_NOISE | SampleType.NOISE | SampleType.CHANGE_IN_SIGNAL_QUALITY:
                            
                            for w in range(0, len(record_windows)):
                                current_window = record_windows[w]
                                window_start = current_window['start']
                                window_end = current_window['end']
                                
                                if window_start <= sample_pos <= window_end:
                                    current_window['tag_positions'].append(sample_pos)
                                    current_window['tag'].append(beat_type)
                                    
                        # Se il tipo di battito non è valido, ignora
                        case _:
                            print(f"Annotazione ignorata: {beat_type}.")
                            continue

                #========================================================================#
                # CACOLO DEL BPM DELLA FINESTRA
                #========================================================================#                       
                for current_window in (record_windows):
                    bpm: int = 0
                    
                    #current_window['BPM'] = current_window['beat_number'] * ((60*self.sample_rate)/self.samples_per_window) 
                    
                    # Calcola il BPM come media della distanza tra i diversi beat (RR interval)
                    beat_times = current_window['beat_time']
                    if len(beat_times) > 1:
                        rr_intervals = [beat_times[i+1] - beat_times[i] for i in range(len(beat_times)-1)]
                        mean_rr = np.mean(rr_intervals)
                        if mean_rr > 0:
                            bpm = 60.0 / mean_rr
                        else:
                            bpm = 0
                    elif len(beat_times) == 1:
                        print(f"Finestra con un solo BPM in {current_window['record_name']}")
                        bpm = 0  # Solo un beat, impossibile calcolare BPM
                    else:
                        bpm = 0  # Nessun beat nella finestra
                    
                    #print(f"Finestra {i} BPM: {current_window['BPM']}")
                    
                    if bpm > MITBIHDataset._MAX_BPM:
                        bpm = MITBIHDataset._MAX_BPM

                    bpm = int(bpm)
                    BPM_value[bpm] += 1
                    
                    current_window['BPM'] = torch.Tensor([bpm]).to(torch.float32)
                    # self._windows[windows_counter] = current_window
                    # windows_counter += 1
                    
                    if self._BPM_toWindows.get(bpm) is None:
                        self._BPM_toWindows[bpm] = [current_window]
                    else:
                        self._BPM_toWindows[bpm].append(current_window)
            
                #associso al record le sue finestre
                self._record_windows[record_name] = record_windows
            
          
          
            if self._mode != DatasetMode.TRAINING:
                windows_counter = 0
                for record_name in self._record_list:
                    for w in self._record_windows[record_name]:
                        self._windows[windows_counter] = w
                        windows_counter += 1
                return
            
            #========================================================================#
            # SOVRACAMPIONAMENTO
            #========================================================================# 
            
            print("\nApplying oversampling...")
            # 1. Trova il valore di BPM più frequente
            # Using BPM_value which counts occurrences of each BPM
            most_common_bpm_val = 0
            max_count = 0
            for bpm_key, count in BPM_value.items():
                if count > max_count:
                    max_count = count
                    most_common_bpm_val = bpm_key

            print(f"Most frequent BPM value: {most_common_bpm_val} with {max_count} occurrences.")

            # 2. Determina la quantità di finestre da replicare per bilanciare
            # We aim to bring other classes closer to the most_common_bpm_val count.
            # A simple strategy is to make all classes at least have `max_count` windows
            # or a multiple of `max_count` (e.g., `max_count * oversample_factor`).

            #target_count_for_oversampling = int(max_count * self.oversample_factor)
            #print(f"Target count for oversampling (max_count * oversample_factor): {target_count_for_oversampling}")
            target_count_for_oversampling = max_count

            oversampled_windows = []
            for bpm_key, windows_list in self._BPM_toWindows.items():
                if windows_list is None or len(windows_list) == 0:
                    continue
                
                # if bpm_key == most_common_bpm_val:
                #     oversampled_windows.extend(windows_list)
                #     continue

                current_count = len(windows_list)
                if current_count < target_count_for_oversampling:
                    num_to_replicate = target_count_for_oversampling - current_count
                    print(f"Oversampling BPM {bpm_key}: replicating {num_to_replicate} windows (current: {current_count})")
                    # Randomly sample with replacement from the current windows_list
                    replicated_windows = random.choices(windows_list, k=num_to_replicate)
                    oversampled_windows.extend(windows_list)
                    oversampled_windows.extend(replicated_windows)
                else:
                    oversampled_windows.extend(windows_list)
            BPM_value = {n: 0 for n in range(0,261)}
            # Aggiorna self._windows con le finestre campionate e sovra-campionate
            self._windows.clear() # Clear existing windows to replace with oversampled ones
            for i, window in enumerate(oversampled_windows):
                self._windows[i] = window
                #BPM_value[int(window['BPM']*260)] += 1
                
            # for i in range(0, 261):
            #     print(f"{i} -> {BPM_value[i]}")
                
                
            print(f"Total windows after oversampling: {len(self._windows)}")
        
            # #========================================================================#
            # # CACOLO DEI PESI
            # #========================================================================# 
            # all_bpms = [self._windows[win_idx]['BPM'].item()*MITBIHDataset._MAX_BPM for win_idx in self._windows] # Assicurati che siano numeri Python standard


            # min_bpm_val = MITBIHDataset._MIN_BPM
            # max_bpm_val = MITBIHDataset._MAX_BPM
            # bpm_bins_range = 5
            
            # num_bins = int((max_bpm_val - min_bpm_val)/bpm_bins_range)
            # bins = np.linspace(min_bpm_val, max_bpm_val, num_bins + 1)
            # bin_indices = np.digitize(all_bpms, bins)
            
            # # Conta le occorrenze in ogni bin
            # # Non tutte i bin avranno campioni, quindi dobbiamo assicurarci di avere un conteggio per tutti i bin
            # unique_bins, counts = np.unique(bin_indices, return_counts=True)
            
            # # Mappa i conteggi ai bin effettivi
            # # Inizializza tutti i conteggi a 0
            # full_counts = np.zeros(num_bins + 2) # +2 per i valori fuori dai bins (inf, -inf)
            # for u_bin, count in zip(unique_bins, counts):
            #     if 0 <= u_bin < len(full_counts): # Evita errori di indice per valori fuori range
            #         full_counts[u_bin] = count
            
            # # Calcola i pesi inversamente proporzionali alla frequenza
            # # Puoi filtrare i bin che non hanno conteggi validi se lo desideri
            # # Ad esempio, considera solo i bin da 1 a num_bins (escludendo -inf e +inf)
            # relevant_counts = full_counts[1:num_bins+1] # Assumendo che bin_indices 0 sia < bins[0] e l'ultimo sia >= bins[ultimo]

            # # Pesi per ogni bin
            # # Aggiungi un piccolo valore per evitare divisioni per zero
            # # Puoi sperimentare diverse formule di weighting (es. 1/count, sqrt(1/count), log(1/count))
            # weights_per_bin = 1.0 / (relevant_counts + 1e-6)

            # # Normalizza i pesi, se vuoi che la loro somma sia 1
            # #weights_per_bin = weights_per_bin / np.sum(weights_per_bin)
            

            
            # self.bpm_bins = torch.tensor(bins/MITBIHDataset._MAX_BPM) # Salva i bordi dei bin
            # self.bin_weights = torch.tensor(weights_per_bin, dtype=torch.float32) # Salva i pesi per i bin
            # print(self.bpm_bins)
            # print(self.bin_weights)
            

          
        except Exception as e:
            print(f"Errore durante il caricamento dei dati per il record {current_record_name}: {e}")
            raise e
            
    
    def _load_data_for_Beat_classification(self):
        
        window_counter: int = 0
        
        for record_name in self._record_list:
            annotations = self._signals_annotations_dict[record_name]
            
            for annotation in annotations:
                beat_type = annotation["annotationotation"]
                sample_pos = annotation["sample_pos"]
                time = annotation["time"]
                
                #Verifico il tipo di annotazione
                if not SampleType.isBeat(beat_type):
                    continue
                
                start = sample_pos - int(self._samples_per_window / 2)
                end = sample_pos + int(self._samples_per_window / 2)
                
                self._windows[window_counter] = {
                    "signal_fragment" : self._signals_dict[record_name][:, start:end],
                    "beatType": beat_type
                }
                window_counter += 1
                

           
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


    def get(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._windows[idx]


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
        
        window_data = self._windows[idx]
        return window_data['signal'], window_data['BPM']


    def plot_windows(self, idx: int, asfile: bool = False, save: bool = False) -> Image.Image | None:
        """
        Plotta una finestra del segnale ECG dato l'indice.
        """
        window = self.get(idx)
        signal = window['signal']
        record_name = window['record_name']
        start = window['start']
        end = window['end']
        beat_positions = window['beat_positions']
        
        x = np.arange(start, end)

        # --- Plot del segnale ---
        output_dir = os.path.join(MITBIHDataset._DATASET_PATH, "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"{record_name}_window_{idx}_ecg_plot.png")

        plt.figure(figsize=(12, 4))
        #plt.xlim(start, end)
        plt.ylim(0, 1)
        
        # Se il segnale ha più canali, plottiamo solo il primo
        plt.plot(x, signal[0].numpy(), label='Segnale ECG MLII')
        plt.plot(x, signal[1].numpy(), label='Segnale ECG V1')
        
        # ax = plt.gca()  # Ottieni l'oggetto Axes corrente
        # for label in ax.get_xticklabels():
        #     x, y = label.get_position()
        #     label.set_position((x + start, y))
            #label.set_x(label.get_position()[0] + start)  # offset negativo = più in basso
        
        plt.title(f"Segnale ECG - Record {record_name} - Finestra {idx} ({start}-{end})")
        #plt.xlabel("Campioni nella finestra")
        plt.ylabel("Ampiezza normalizzata")
        plt.legend()
        plt.grid()
        
        
        # Aggiungi linee verticali e annotazioni per i beat
        for p, label in zip(window['beat_positions'], window['beat_labels']):
            plt.axvline(x=p, color='red', linestyle='--', linewidth=0.8)
            plt.text(p-1, 0.95, str(label.value), color='red', rotation=90, fontsize=8, ha='center', va='top')

        # Aggiungi linee verticali e annotazioni per i tag
        for p, tag in zip(window['tag_positions'], window['tag']):
            plt.axvline(x=p, color='blue', linestyle=':', linewidth=0.8)
            plt.text(p-1, 0.90, str(tag.value), color='blue', rotation=0, fontsize=8, ha='center', va='bottom')

        # Mostra il valore del BPM sotto il grafico
        plt.figtext(0.5, 0.01, f"BPM: {window['BPM'].item():.2f}", ha='center', fontsize=10, color='green')
        
        if save:
            plt.savefig(output_filepath)
            plt.close()
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            return img
      
       
    def plot_all_windows_for_record(self, record_name: str, output_dir:str):
        """
        Plotta tutte le finestre di un record in modo continuo.
        """
        if record_name not in self._record_windows:
            print(f"Record {record_name} non trovato in self._file_windows.")
            return

        windows = self._record_windows[record_name]
        num_windows = len(windows)
        if num_windows == 0:
            print(f"Nessuna finestra trovata per il record {record_name}.")
            return

        #output_dir = os.path.join(MITBIHDataset._DATASET_PATH, "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"{record_name}_all_windows_ecg_plot.png")

    
        
        # Plotta ogni finestra e concatena le immagini verticalmente
        window_imgs = []
        
        
        for i in tqdm(range(num_windows), desc=f"Plotting windows for record {record_name}"):
            img = self.plot_windows(idx=i + sum(len(self._record_windows[r]) for r in self._record_list if r < record_name), asfile=False, save=False)
            if img is not None:
                window_imgs.append(img.convert("RGB"))

        if not window_imgs:
            print(f"Nessuna immagine generata per le finestre di {record_name}.")
            return

        # Calcola la larghezza massima e l'altezza totale
        widths, heights = zip(*(img.size for img in window_imgs))
        max_width = max(widths)
        total_height = sum(heights)

        # Crea una nuova immagine vuota per concatenare tutte le finestre
        concatenated_img = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in window_imgs:
            concatenated_img.paste(img, (0, y_offset))
            y_offset += img.size[1]

        concatenated_img.save(output_filepath)
        print(f"Immagine concatenata salvata in: {output_filepath}")
        
        
        # plt.savefig(output_filepath)
        # plt.close()
        
    


    def plot_bpm_distribution(self, output_filepath: str):
        """
        Realizza e salva un diagramma della distribuzione dei valori di BPM
        presenti nelle finestre del dataset, evidenziando il valore più frequente
        e elencando i file per ogni intervallo di BPM sotto il grafico.

        Args:
            output_filepath (str): Il percorso completo dove salvare l'immagine del grafico.
        """
        if not self._windows:
            print(f"Nessuna finestra caricata per la modalità {self._mode.name}.")
            return

        all_bpms = []
        # Dizionario per raccogliere i record per intervallo di BPM
        records_by_bpm_bin = {}

        for window_data in self._windows.values():
            # Converti il tensore a scalare Python e aggiungi alla lista
            bpm = window_data['BPM'].item()
            record_name = window_data['record_name']
            all_bpms.append(bpm)

            # Temporaneamente, raccogliamo solo i BPM. Li assoceremo ai bin dopo aver calcolato l'istogramma
            # per garantire che usiamo gli stessi bin per il plot e per l'elenco.


        if not all_bpms:
            print("Nessun valore di BPM trovato.")
            return

        all_bpms_np = np.array(all_bpms)

        # Calcola l'istogramma e i bin
        # Usa density=False per ottenere i conteggi effettivi
        counts, bin_edges = np.histogram(all_bpms_np, bins='auto', density=False)

        # Trova l'indice del bin con la frequenza massima
        max_count_index = np.argmax(counts)

        # Trova il valore di BPM più frequente (centro del bin con max count)
        mode_bpm_value = (bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2

        # Ora associamo i record ai bin basati sui bin_edges calcolati
        for window_data in self._windows.values():
            bpm = window_data['BPM'].item()
            record_name = window_data['record_name']
            # Trova l'indice del bin a cui appartiene questo BPM.
            # np.digitize restituisce l'indice del bin *alla destra* del valore,
            # eccetto per il bordo destro dell'ultimo bin.
            bin_index = np.digitize(bpm, bin_edges) - 1 # -1 per allineare con gli indici di `counts` e `bin_edges`

            # Assicurati che l'indice sia valido (0 <= bin_index < len(counts))
            if 0 <= bin_index < len(counts):
                bin_range_key = (bin_edges[bin_index], bin_edges[bin_index + 1])
                if bin_range_key not in records_by_bpm_bin:
                    records_by_bpm_bin[bin_range_key] = set() # Usa un set per evitare duplicati
                records_by_bpm_bin[bin_range_key].add(record_name)

        # Ordina gli intervalli di BPM per una migliore leggibilità
        sorted_bpm_bins = sorted(records_by_bpm_bin.keys())

        # Prepara il testo per l'elenco dei file
        text_list = ["Files per intervallo di BPM:"]
        for bin_range in sorted_bpm_bins:
            records = sorted(list(records_by_bpm_bin[bin_range])) # Ordina i nomi dei record
            # Formatta l'intervallo del bin
            if bin_range == sorted_bpm_bins[-1]: # Ultimo bin, usa [] per includere il bordo superiore
                bin_text = f"  [{bin_range[0]:.2f} - {bin_range[1]:.2f}] BPM: {', '.join(records)}"
            else: # Altri bin, usa [)
                bin_text = f"  [{bin_range[0]:.2f} - {bin_range[1]:.2f}) BPM: {', '.join(records)}"
            text_list.append(bin_text)

        file_list_text = "\n".join(text_list)

        # Stima l'altezza necessaria per il testo basata sul numero di righe e sulla dimensione del font.
        # Questa è una stima grezza e potrebbe richiedere aggiustamenti.
        text_line_height_inches = 0.15 # Stima altezza per riga in pollici per fontsize 9
        num_text_lines = len(text_list)
        required_text_height_inches = num_text_lines * text_line_height_inches
        text_buffer_inches = 0.5 # Buffer aggiuntivo sotto il testo per spaziatura

        # Altezza desiderata per l'area principale del grafico in pollici
        desired_plot_height_inches = 6
        figure_width_inches = 12

        # Calcola l'altezza totale della figura necessaria
        total_figure_height_inches = desired_plot_height_inches + required_text_height_inches + text_buffer_inches

        # Crea la figura con l'altezza totale calcolata
        fig, ax = plt.subplots(figsize=(figure_width_inches, total_figure_height_inches))

        # Plotta l'istogramma sugli assi principali 'ax'
        ax.hist(all_bpms_np, bins=bin_edges, edgecolor='black', alpha=0.7)

        # Aggiungi marker per il valore più frequente
        modal_bin_height = counts[max_count_index]
        ax.plot(mode_bpm_value, modal_bin_height, 'r*', markersize=15, label=f'Valore più frequente: {mode_bpm_value:.2f}')

        ax.set_title(f"Distribuzione dei valori di BPM per la modalità {self._mode.name}")
        ax.set_xlabel("Battiti per minuto (BPM)")
        ax.set_ylabel("Frequenza")
        ax.legend()
        ax.grid(axis='y', alpha=0.75)

        # Calcola il parametro 'bottom' per subplots_adjust.
        # Questo è il rapporto tra lo spazio necessario per il testo (più buffer)
        # e l'altezza totale della figura.
        bottom_margin_fraction = (required_text_height_inches + text_buffer_inches) / total_figure_height_inches

        # Aggiusta i parametri del subplot per fare spazio al testo nella parte inferiore.
        # Il parametro 'bottom' è la posizione del bordo inferiore dei subplot,
        # relativa al bordo inferiore della figura (0=fondo, 1=cima).
        # Imposta anche 'top' per definire lo spazio sopra il grafico.
        fig.subplots_adjust(bottom=bottom_margin_fraction, top=0.95)

        # Aggiungi il testo sotto il grafico.
        # Posiziona il testo partendo dal bordo sinistro (0.02) e leggermente sopra
        # il bordo inferiore della figura (0.01).
        # 'transform=fig.transFigure' posiziona il testo rispetto all'intera figura
        # (0,0 è l'angolo in basso a sinistra, 1,1 è in alto a destra).
        # 'va='bottom'' allinea il bordo inferiore del blocco di testo alla posizione y.
        fig.text(0.02, 0.01, file_list_text, fontsize=9, va='bottom', ha='left', wrap=True)


        # Assicurati che la directory di output esista
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Creata directory: {output_dir}")

        # Salva il grafico
        plt.savefig(output_filepath)
        plt.close(fig) # Chiudi la figura per liberare memoria


        print(f"Diagramma della distribuzione dei BPM con elenco file salvato in: {output_filepath}")


