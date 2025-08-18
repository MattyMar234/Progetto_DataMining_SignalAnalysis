from collections import Counter
import hashlib
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
from imblearn.over_sampling import SMOTE


from sklearn.utils.class_weight import compute_class_weight

from setting import APP_LOGGER

# Definizione delle modalità del dataset
class DatasetMode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    
class DatasetDataMode(Enum):
    BEAT_CLASSIFICATION = auto()
    BPM_REGRESSION = auto()
    
class DatasetChannels(Enum):
    ONE = 1
    TWO = 2

#https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#leads
class BeatType(Enum):
    
    UNKNOWN_BEAT_Q = "QB", 0, "Unknown beat"
    #LEFT_OR_RIGHT_BUNDLE_BRANCH_BLOCK_BEAT = "B",-1, "?"
    
    
    #Normal e battiti di conduzione normale
    NORMAL_BEAT = "N",1, "Normal beat"
    LEFT_BUNDLE_BRANCH_BLOCK_BEAT = "L",2, "Left bundle branch block beat"
    RIGHT_BUNDLE_BRANCH_BLOCK_BEAT = "R",3, "Right bundle branch block beat"
    
    NODAL_JUNCTIONAL_ESCAPE_BEAT = "j",4, "Nodal (junctional) escape beat"
    ATRIAL_ESCAPE_BEAT = "e",5, "Atrial escape beat"
    
    #Battiti sopra-ventricolari (Supraventricular ectopic)
    ATRIAL_PREMATURE_BEAT = "A",6, "Atrial premature beat"
    ABERRATED_ATRIAL_PREMATURE_BEAT = "a",7, "Aberrated atrial premature beat"
    NODAL_JUNCTIONAL_PREMATURE_BEAT = "J",8, "Nodal (junctional) premature beat"
    SUPRAVENTRICULAR_PREMATURE_BEAT = "S",9, "Supraventricular premature beat"
    
    #Battiti ventricolari (Ventricular ectopic)
    PREMATURE_VENTRICULAR_CONTRACTION = "V", 10, "Premature ventricular contraction"
    VENTRICULAR_ESCAPE_BEAT = "E",11, "Ventricular escape beat"
    
    
    #Fusion beats
    FUSION_OF_VENTRICULAR_AND_NORMAL_BEAT = "F", 12, "Fusion of ventricular and normal beat"
    FUSION_OF_PACED_AND_NORMAL_BEAT = "f", 13, "Fusion of paced and normal beat"
   
    #others
    VENTRICULAR_FLUTTER_WAVE = "!", 14, "Ventricular flutter wave"
    PACED_BEAT_SLASH = "/", 15, "Paced beat"
    PACED_BEAT_P = "P", 15, "Paced beat"
    ISOLATED_QRS_LIKE_ARTIFACT = "|", 16, "Isolated QRS-like artifact"
    
  
    #ANNOTATION
    CHANGE_IN_SIGNAL_QUALITY = "~", -1, "?"
    NOISE = "X x", -1, "?"
    START_NOISE = "[", -1, "?"
    END_NOISE = "]", -1, "?"
    START_SEG_PLUS = "+", -1, "?"
    COMMENT = "\"", -1, "?"
    

    
    def __str__(self):
        return self.value[0]
    

    @classmethod
    def toList(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def tokenize(cls, type: str | int) -> 'BeatType':
       
        # Mappa il carattere o l'intero in un oggetto SampleType usando i valori degli enum
        if isinstance(type, int):
            for member in cls:
                if member.value[1] == type:
                    return member
            raise ValueError(f"Tipo di battito sconosciuto: {type}. Non può essere convertito in SampleType.")
        
        elif isinstance(type, str):
            # if len(type) != 1:
            #     raise ValueError(f"Tipo di battito non valido: {type}")
        
            type_char = type.strip()[0]
            for member in cls:
                if type_char in member.value[0]:
                    return member
            raise ValueError(f"Tipo di battito sconosciuto: {type_char}. Non può essere convertito in SampleType.")
        else:
            raise ValueError(f"Tipo di battito non valido: {type}. Tipo del dato: {type(type)}. Deve essere un carattere o un intero.")

    @classmethod  
    def isBeat(cls, annotation: 'BeatType') -> bool:
        return annotation in {
            cls.NORMAL_BEAT, cls.LEFT_BUNDLE_BRANCH_BLOCK_BEAT, cls.RIGHT_BUNDLE_BRANCH_BLOCK_BEAT,
            #cls.LEFT_OR_RIGHT_BUNDLE_BRANCH_BLOCK_BEAT, 
            cls.ISOLATED_QRS_LIKE_ARTIFACT,
            cls.ATRIAL_PREMATURE_BEAT, cls.ABERRATED_ATRIAL_PREMATURE_BEAT,
            cls.NODAL_JUNCTIONAL_PREMATURE_BEAT, cls.NODAL_JUNCTIONAL_ESCAPE_BEAT,
            cls.SUPRAVENTRICULAR_PREMATURE_BEAT, cls.PREMATURE_VENTRICULAR_CONTRACTION,
            cls.VENTRICULAR_ESCAPE_BEAT, cls.ATRIAL_ESCAPE_BEAT,
            cls.FUSION_OF_VENTRICULAR_AND_NORMAL_BEAT, cls.FUSION_OF_PACED_AND_NORMAL_BEAT,
            cls.VENTRICULAR_FLUTTER_WAVE, cls.UNKNOWN_BEAT_Q, cls.PACED_BEAT_SLASH
        }
    
    @classmethod
    def isTag(cls, annotation: 'BeatType') -> bool:
        return annotation in {
            cls.CHANGE_IN_SIGNAL_QUALITY, cls.NOISE, cls.START_NOISE,
            cls.END_NOISE, cls.START_SEG_PLUS, cls.COMMENT
        }
        
    @classmethod
    def getBeatCategory(cls, beat: 'BeatType') -> int:
        match beat:
            
            #N
            case cls.NORMAL_BEAT | \
                 cls.LEFT_BUNDLE_BRANCH_BLOCK_BEAT | \
                 cls.RIGHT_BUNDLE_BRANCH_BLOCK_BEAT | \
                 cls.ATRIAL_ESCAPE_BEAT | \
                 cls.NODAL_JUNCTIONAL_ESCAPE_BEAT:
                return 0
            
            #SVEB
            case cls.ATRIAL_PREMATURE_BEAT | \
                 cls.ABERRATED_ATRIAL_PREMATURE_BEAT | \
                 cls.NODAL_JUNCTIONAL_PREMATURE_BEAT | \
                 cls.SUPRAVENTRICULAR_PREMATURE_BEAT:
                return 1
            
            #VEB
            case cls.PREMATURE_VENTRICULAR_CONTRACTION | \
                 cls.VENTRICULAR_ESCAPE_BEAT:
                return 2
            
            #F
            case cls.FUSION_OF_VENTRICULAR_AND_NORMAL_BEAT:
                return 3
            
            #Q
            case cls.UNKNOWN_BEAT_Q | \
                 cls.FUSION_OF_PACED_AND_NORMAL_BEAT | \
                 cls.ISOLATED_QRS_LIKE_ARTIFACT | \
                 cls.PACED_BEAT_SLASH | \
                 cls.VENTRICULAR_FLUTTER_WAVE: 
                return 4
            case _:
                raise ValueError(f"Il beat {beat} non ha una mappatura numerica definita.")

    @classmethod
    def mapBeatClass_to_Label(cls, idx: int) -> str:
        if idx <= 0 or idx >= cls.num_classes(): return "Unknow"
        
        for enm in cls.toList():
            if enm[1]==idx: return f"{enm[2]} ({enm[0]})"
        else:
            return "Unknow"
        
    @classmethod
    def mapBeatCategory_to_Label(cls, idx: int) -> str:
        match idx:
            case 0: return "Normal"
            case 1: return  "SVEB"
            case 2: return  "VEB"
            case 3: return  "Fusion"
            case _: return  "Unclassifiable"
        
        
    @classmethod
    def getBeatClass(cls, beat: 'BeatType') -> int:
        return beat.value[1]
    
    @classmethod
    def num_classes(cls) -> int:
        """Restituisce il numero di classi per la classificazione (escluse le annotazioni e i battiti ignorati)."""
        return 17
    
    @classmethod
    def num_of_category(cls) -> int:
        """Restituisce il numero di classi per la classificazione (escluse le annotazioni e i battiti ignorati)."""
        return 5

    @classmethod
    def get_ignore_class_value(cls) -> int:
        """Restituisce l'etichetta numerica che dovrebbe essere ignorata (es. UNKNOWN)"""
        return 0
    
    @classmethod
    def get_ignore_category_value(cls) -> int:
        """Restituisce l'etichetta numerica che dovrebbe essere ignorata (es. UNKNOWN)"""
        return 4
      
    
  
class MITBIHDataset(Dataset):
    """
    Classe PyTorch Dataset adattata per il database MIT-BIH Arrhythmia
    con dati in formato CSV per segnali e TXT per annotazioni.
    """
    
    _RECORDS_MLII_V1 = ['101','105','106', '107', '108', '109','111', '112','113','115', '116','118','119','121','122',
                        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212','213', '214','215','217',
                        '219', '220', '221', '222', '223', '228', '230','231','232','233','234']
    _RECORDS_MLII_V2 = ['117', '103']
    _RECORDS_MLII_V4 = ['124']
    _RECORDS_MLII_V5 = ['100','114','123']
    _RECORDS_V5_V2   = ['104', '102']
    
    _MLII_COL: str = 'MLII'
    _V1_COL: str = 'V1'
    _V2_COL: str = 'V2'
    _V3_COL: str = 'V3'
    _V4_COL: str = 'V4'
    _V5_COL: str = 'V5'
    
    # lista dei record da utilizzare
    _ALL_RECORDS: Final[list] = _RECORDS_MLII_V1 #+ _RECORDS_MLII_V2 + _RECORDS_MLII_V4 + _RECORDS_MLII_V5# + _RECORDS_V5_V2
    
    _FILES_CHEKED: bool = False
    _DATASET_PATH: None | str = None
    _RISOLUZIONE_ADC: int = 11
    _MIN_VALUE: int = 0
    _MAX_VALUE: int = 2**_RISOLUZIONE_ADC - 1
    _MAX_SAMPLE_NUM: int = 650000
    _MAX_BPM=260
    _MIN_BPM=0
    
    __SAMPLE_RATE: int | None = None
    __RANDOM_SEED: int | None = None
    __USE_SMOTE: bool = True
    # Dizionario per memorizzare il segnale di ogni record
    _ALL_SIGNALS_DICT: Dict[str, torch.Tensor] = {}
    _WINDOWS: Dict[int, Dict[str, Any]] = {}
    
    # Dizionario per memorizzare le annotazioni di ogni segnale
    _ALL_SIGNALS_BEAT_ANNOTATIONS_DICT: Dict[str, List[Dict[str, Any]]] = {}
    _ALL_SIGNALS_TAG_ANNOTATIONS_DICT: Dict[str, List[Dict[str, Any]]] = {}
    

    @classmethod
    def resetDataset(cls) -> None:
        cls._DATASET_PATH = None
        cls._FILES_CHEKED = False
        cls._ALL_SIGNALS_DICT.clear()
        cls._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT.clear()
        cls._ALL_SIGNALS_TAG_ANNOTATIONS_DICT.clear()

    @classmethod
    def initDataset(cls, path: str, sample_rate: int = 360, *, fill_and_concat_missing_channels: bool = False, random_seed: int = 42):
        """
        Imposta il percorso del dataset e carica staticamente tutti i dati.
        Questo metodo dovrebbe essere chiamato una volta all'inizio del programma.
        """
        if cls._FILES_CHEKED and cls._DATASET_PATH == path:
            APP_LOGGER.info(f"Il percorso del dataset è già impostato su: {cls._DATASET_PATH}")
            return
        
        cls.__SAMPLE_RATE = sample_rate
        cls.__RANDOM_SEED = random_seed
        cls._DATASET_PATH = path
        
        #========================================================================#
        # VERIFICO I FILES
        #========================================================================#
        if not os.path.isdir(path):
            raise FileNotFoundError(f"La directory specificata non esiste: {path}")
        APP_LOGGER.info(f"Percorso del dataset impostato su: {cls._DATASET_PATH}")
        
        APP_LOGGER.info("Verifica dei files")
        for record_name in cls._ALL_RECORDS:
            csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
            txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")
            assert os.path.exists(csv_filepath), f"file {csv_filepath} non trovato" 
            assert os.path.exists(txt_filepath), f"file {txt_filepath} non trovato" 
        
        APP_LOGGER.info("Completato")
        
        cls._FILES_CHEKED = True
        
        #========================================================================#
        # CARICO I DATI
        #========================================================================#
        APP_LOGGER.info("caricamneto dei dati ")
        cls.__load_signals(fill_and_concat_missing_channels=fill_and_concat_missing_channels)
        APP_LOGGER.info("Completato")
        
        #========================================================================#
        # NORMALIZZAZIONE DEI DATI
        #========================================================================#
        APP_LOGGER.info("Normalizzazione dei dati")
        # min_list = []
        # max_list = []
        
        # #cerco i massimi e i minimi di ogni segnale
        # for record_name in cls._ALL_RECORDS:
        #     signal = cls._ALL_SIGNALS_DICT[record_name]
        #     max_list.append(torch.max(signal).item())
        #     min_list.append(torch.min(signal).item())
 
        # #cerco il massimo e il minimo assoluto
        # cls._MAX_VALUE = max(max_list)
        # cls._MIN_VALUE = min(min_list)
        
        
        #normalizzo tutti i segnali
        for record_name in cls._ALL_RECORDS:
            signal = cls._ALL_SIGNALS_DICT[record_name]
            signal = (signal - MITBIHDataset._MIN_VALUE) / (MITBIHDataset._MAX_VALUE - MITBIHDataset._MIN_VALUE) 
            cls._ALL_SIGNALS_DICT[record_name] = signal
        
        APP_LOGGER.info("Completata")
        APP_LOGGER.info(f"Valore massimo trovato: {cls._MAX_VALUE}")
        APP_LOGGER.info(f"Valore minimo trovato: {cls._MIN_VALUE}")
        
    @classmethod
    def __load_signals(cls, fill_and_concat_missing_channels: bool = False) -> None:
        
        colums_list: list = [
            MITBIHDataset._MLII_COL,
            MITBIHDataset._V1_COL,
            MITBIHDataset._V2_COL,
            #MITBIHDataset._V3_COL, non presente nel dataset
            MITBIHDataset._V4_COL,
            MITBIHDataset._V5_COL
        ]
        
        progress_bar = tqdm(total=len(cls._ALL_RECORDS), desc="Caricamento Records")
        
        try:    
            for record_name in cls._ALL_RECORDS:
                progress_bar.set_description(f"record {record_name}")
                
                #progress_bar.display(record_name)
                csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
                txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")
                signal: torch.Tensor | None = None
        
                
                #========================================================================#
                # ESTRAGGO IL SEGNALE
                #========================================================================#
                signal_list = []
                
                # --- Leggi il segnale dal file CSV ---
                df = pd.read_csv(csv_filepath)
                
                for idx, col in enumerate(df.columns):
                    df = df.rename(columns={df.columns[idx]: df.columns[idx].replace('\'', '')})

                for col in colums_list:
                    if col in df.columns:
                        data = torch.from_numpy(df[col].values).unsqueeze(0)
                    elif fill_and_concat_missing_channels:
                        data = torch.zeros(MITBIHDataset._MAX_SAMPLE_NUM).unsqueeze(0)
                    else:
                        continue
    
                    signal_list.append(data)

                signal = torch.cat(signal_list, dim=0)
                
                # Salva il segnale per il record corrente
                cls._ALL_SIGNALS_DICT[record_name] = signal.float() 
                
                #========================================================================#
                # ESTRAGGO LE ANNOTAZIONI
                #========================================================================#
                with open(txt_filepath, 'r') as f:
                    f.readline() # Salta la prima riga (header)
                    
                    beat_dataDict_List: List[Dict[str, any]] = []
                    tag_dataDict_List: List[Dict[str, any]] = []
                    
                    for line in f:
                        line = line.strip()  # Rimuovi spazi bianchi
                        parts = line.split() # Dividi la riga in base agli spazi e ignore le stringe vuote

                        # Una riga di annotazione valida dovrebbe avere almeno 3 parti (Time, Sample #, Type)
                        if len(parts) < 3:
                            raise ValueError(f"Riga di annotazione non valida: {line}.")

                        annotationType = BeatType.tokenize(parts[2])

                        if BeatType.isBeat(annotationType):
                            beat_dataDict_List.append({
                                "annotation" : annotationType,          #tipologia di sample
                                "sample_pos" : int(parts[1]),           # Indice del campione
                                "time" : cls._formatTime(parts[0])      # Tempo in secondi
                            })
                        elif BeatType.isTag(annotationType):
                            tag_dataDict_List.append({
                                "annotation" : annotationType,          #tipologia di sample
                                "sample_pos" : int(parts[1]),           # Indice del campione
                                "time" : cls._formatTime(parts[0])      # Tempo in secondi
                            })
                            
                                            
                # Salva le annotazioni per il record corrente
                cls._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_name] = beat_dataDict_List 
                cls._ALL_SIGNALS_TAG_ANNOTATIONS_DICT[record_name] = tag_dataDict_List
                progress_bar.update(1)
        finally:
            progress_bar.close()

    def __new__(cls, *args, **kwargs):
        return super(MITBIHDataset, cls).__new__(cls)  
        
    
    def __init__(
            self, 
            *,
            sample_per_window: int | None = None, 
            sample_per_stride: int | None = None, 
            mode: DatasetMode | None = None, 
            dataMode: DatasetDataMode | None = None, 
            channels: DatasetChannels = DatasetChannels.TWO
        ):
        assert MITBIHDataset._DATASET_PATH, "Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo."
        assert MITBIHDataset._FILES_CHEKED, "files del dataset non verificati"
        
   
        self._mode: DatasetMode = mode
        self._dataMode: DatasetDataMode = dataMode
        self._channels: DatasetChannels = channels
        
        self._samples_per_window: int = sample_per_window
        self._samples_per_side:int = sample_per_stride
        
  
        #dizionario delle finestre di dati     
        self._windows: Dict[int, Dict[str, any]] = {} 
        self._class_weights: torch.Tensor | None = None  
        self._category_weights: torch.Tensor | None = None      
        self._record_windows: Dict[str, Dict[int, Dict[str, any]]] = {}
        self._BPM_toWindows: Dict[int, list] = {}
        
        train_ratio = 0.75
        test_ratio = 0.25
        # val_ratio = 0.1
        # test_ratio = 0.1
        
        APP_LOGGER.info("-"*100)
        APP_LOGGER.info(f"Caricamento valori per {self._mode} in modalità {self._dataMode}")
        
        match self._dataMode:
            case DatasetDataMode.BPM_REGRESSION: 
                self._load_data_for_BPM_regression()
                
            case DatasetDataMode.BEAT_CLASSIFICATION: 
                
                total_beat_number: int = 0
                y: List[int] = []
                
                # for record_key in MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT.keys():
                #     for beat in MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_key]:
                #         y.append(BeatType.getBeatClass(beat["annotation"]))
                    
                #     total_beat_number += len(MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_key])
                
                # x = range(0, total_beat_number)
                
                # #train_indices, temp_indices = train_test_split(index_list, test_size=(val_ratio + test_ratio), random_state=MITBIHDataset.__RANDOM_SEED)
                # #val_indices, test_indices = train_test_split(temp_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=MITBIHDataset.__RANDOM_SEED)

                # #mi interessano solo gli indici
                # train_indices, test_indices, _, _ = train_test_split(
                #     x,y,
                #     stratify=y,
                #     test_size=test_ratio, 
                #     random_state=MITBIHDataset.__RANDOM_SEED
                # )
                
                # Costruzione della lista target y
                for record_key in MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT.keys():
                    for beat in MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_key]:
                        y.append(BeatType.getBeatClass(beat["annotation"]))
                    total_beat_number += len(MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_key])

                x = np.arange(total_beat_number)
                y = np.array(y)

                # Conta le occorrenze di ogni classe
                class_counts = Counter(y)
                rare_classes = [cls for cls, count in class_counts.items() if count < 8]

                print(f"Classi rare trovate: {rare_classes}")

                train_indices = []
                val_indices = []

                for rare_class in rare_classes:
                    indices = np.where(y == rare_class)[0]
                    if len(indices) == 1:
                        # La classe ha un solo elemento: va nel train
                        train_indices.append(indices[0])
                    else:
                        # Metti uno nel train e uno nel test
                        train_indices.append(indices[0])
                        val_indices.append(indices[1])

                # Questi sono gli indici rari già assegnati
                rare_assigned = np.array(train_indices + val_indices)
                remaining_indices = np.setdiff1d(x, rare_assigned)

                # Ora split dei rimanenti dati (stratificato)
                remaining_y = y[remaining_indices]

                # Controlla se ci sono ancora problemi di classi con un solo elemento
                remaining_counts = Counter(remaining_y)
                problematic_classes = [cls for cls, count in remaining_counts.items() if count < 2]

                if problematic_classes:
                    print(f"Attenzione: ancora classi troppo piccole per stratify: {problematic_classes}")
                    # Sposta queste classi tutte nel test
                    for cls in problematic_classes:
                        cls_indices = remaining_indices[np.where(remaining_y == cls)[0]]
                        val_indices.extend(cls_indices.tolist())
                    # Aggiorna remaining
                    rare_assigned = np.array(train_indices + val_indices)
                    remaining_indices = np.setdiff1d(x, rare_assigned)
                    remaining_y = y[remaining_indices]

                # Se ci sono dati sufficienti, fai stratify
                if len(np.unique(remaining_y)) > 1:
                    tmp_train_indices, tmp_val_indices = train_test_split(
                        remaining_indices,
                        stratify=remaining_y,
                        test_size=test_ratio,
                        random_state=MITBIHDataset.__RANDOM_SEED
                    )
                else:
                    # Non possibile stratify se rimane solo una classe
                    tmp_train_indices, tmp_val_indices = train_test_split(
                        remaining_indices,
                        test_size=test_ratio,
                        random_state=MITBIHDataset.__RANDOM_SEED
                    )

                # Aggiungi ai train/test finali
                train_indices = np.concatenate([train_indices, tmp_train_indices])
                test_indices = np.concatenate([val_indices, tmp_val_indices])
                
                # Controllo finale
                print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
                print(f"Classi nel train: {np.unique(y[train_indices])}")
                print(f"Classi nel test: {np.unique(y[test_indices])}")

               
                match self._mode:
                    case DatasetMode.TRAINING:
                        self.__load_data_for_Beat_classification(train_indices)
            
                    #case DatasetMode.VALIDATION:
                        #self.__load_data_for_Beat_classification(val_indices)

                    case DatasetMode.VALIDATION | DatasetMode.TEST:
                        self.__load_data_for_Beat_classification(test_indices)
            
                    case _ :
                        raise ValueError(f"Modalità non valida: {self._mode}") 
            case _ :
                raise ValueError(f"Modalità per i dati non valida: {self._dataMode}")
        
    def getClassWeights(self) -> torch.Tensor:
        assert self._dataMode == DatasetDataMode.BEAT_CLASSIFICATION, "I pesi sono disponibili solo per la classificazione"
        return self._class_weights.detach().clone()
    
    def getCategoryWeights(self) -> torch.Tensor:
        assert self._dataMode == DatasetDataMode.BEAT_CLASSIFICATION, "I pesi sono disponibili solo per la classificazione"
        return self._category_weights.detach().clone()
    
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
                        case BeatType.NORMAL_BEAT | BeatType.LEFT_BUNDLE_BRANCH_BLOCK_BEAT | BeatType.RIGHT_BUNDLE_BRANCH_BLOCK_BEAT | BeatType.LEFT_OR_RIGHT_BUNDLE_BRANCH_BLOCK_BEAT | BeatType.ISOLATED_QRS_LIKE_ARTIFACT\
                            | BeatType.ATRIAL_PREMATURE_BEAT | BeatType.ABERRATED_ATRIAL_PREMATURE_BEAT | BeatType.NODAL_JUNCTIONAL_PREMATURE_BEAT | BeatType.NODAL_JUNCTIONAL_ESCAPE_BEAT | BeatType.SUPRAVENTRICULAR_PREMATURE_BEAT\
                            | BeatType.PREMATURE_VENTRICULAR_CONTRACTION | BeatType.VENTRICULAR_ESCAPE_BEAT | BeatType.ATRIAL_ESCAPE_BEAT \
                            | BeatType.FUSION_OF_VENTRICULAR_AND_NORMAL_BEAT | BeatType.FUSION_OF_PACED_AND_NORMAL_BEAT \
                            | BeatType.UNKNOWN_BEAT_Q | BeatType.PACED_BEAT_SLASH | BeatType.VENTRICULAR_FLUTTER_WAVE:
                                
                                        
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
                        case BeatType.START_SEG_PLUS | BeatType.COMMENT | BeatType.END_NOISE | BeatType.START_NOISE | BeatType.NOISE | BeatType.CHANGE_IN_SIGNAL_QUALITY:
                            
                            for w in range(0, len(record_windows)):
                                current_window = record_windows[w]
                                window_start = current_window['start']
                                window_end = current_window['end']
                                
                                if window_start <= sample_pos <= window_end:
                                    current_window['tag_positions'].append(sample_pos)
                                    current_window['tag'].append(beat_type)
                                    
                        # Se il tipo di battito non è valido, ignora
                        case _:
                            APP_LOGGER.warning(f"Annotazione ignorata: {beat_type}.")
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
                        APP_LOGGER.info(f"Finestra con un solo BPM in {current_window['record_name']}")
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
            
            APP_LOGGER.info("Applying oversampling...")
            # 1. Trova il valore di BPM più frequente
            # Using BPM_value which counts occurrences of each BPM
            most_common_bpm_val = 0
            max_count = 0
            for bpm_key, count in BPM_value.items():
                if count > max_count:
                    max_count = count
                    most_common_bpm_val = bpm_key

            APP_LOGGER.info(f"Most frequent BPM value: {most_common_bpm_val} with {max_count} occurrences.")

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
                    APP_LOGGER.info(f"Oversampling BPM {bpm_key}: replicating {num_to_replicate} windows (current: {current_count})")
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
                
                
            APP_LOGGER.info(f"Total windows after oversampling: {len(self._windows)}")
        
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
            APP_LOGGER.error(f"Errore durante il caricamento dei dati per il record {current_record_name}: {e}")
            raise e
            
    

    def __load_data_for_Beat_classification(self, indices: List[int]):
        indices = sorted(indices)
        
        idx: int = 0
        counter: int = 0
        window_counter: int = 0
        done: bool = False
        
        progress_bar = tqdm(total=len(indices))
        
        dataDict: Dict[BeatType, List[torch.Tensor]] = {}
        
        try:
            for record_name in MITBIHDataset._ALL_RECORDS:
                progress_bar.set_description(f"record {record_name}")
                
                if done:  break
                
                annotations = MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_name]
                signal = MITBIHDataset._ALL_SIGNALS_DICT[record_name]
                
                
                for annotation in annotations:
                    
                    #se ho preso tutti gli elementi
                    if idx >= len(indices):
                        done=True
                        break
                    
                    #se non ho ancora raggiugnto l'indice dell'elemento da utilizzare
                    elif indices[idx] > counter:
                        counter += 1
                        continue
                    
                    #se h oraggiugnto l'indice dell'elemeneto da utilizzare
                    elif indices[idx] == counter:
                        progress_bar.update(1)
                        counter += 1
                        idx+=1
                        
                    
                    
                    beat_type = annotation["annotation"]
                    sample_pos = annotation["sample_pos"]
                    #time = annotation["time"]
                    
                    #Verifico il tipo di annotazione
                    if not BeatType.isBeat(beat_type):
                        continue
                    
                    start = sample_pos - int(self._samples_per_window / 2)
                    end = sample_pos + int(self._samples_per_window / 2)


                    #ignoro il campione se non ci sta nella finestra
                    # if start < 0 or end >= MITBIHDataset._MAX_SAMPLE_NUM:
                    #     continue
                    
                    if start < 0:
                        offset = abs(start)
                        start = 0
                        end = end + offset
                    
                    if end > MITBIHDataset._MAX_SAMPLE_NUM - 1:
                        offset = end - (MITBIHDataset._MAX_SAMPLE_NUM -1)
                        end = MITBIHDataset._MAX_SAMPLE_NUM -1
                        start = start - offset
                      
                    # # Extract and pad signal fragment
                    signal_fragment = signal[:, start:end]
                    
                    if dataDict.get(beat_type) is None:
                        dataDict[beat_type] = [signal_fragment]
                    else:
                        dataDict[beat_type].append(signal_fragment)
                    
        finally:
            progress_bar.close()  
            
        APP_LOGGER.info("Valori di ogni classe:")
        for beat_type, signals in dataDict.items():
            APP_LOGGER.info(f"{beat_type.value[0]}: {len(signals)} segnali")
        
        if self._mode == DatasetMode.TRAINING and MITBIHDataset.__USE_SMOTE:
            APP_LOGGER.info(f"Applicazione della SMOTE")   
            
            
            X = []
            y = []
            
            threshold = 2
            ignored_x = []
            ignored_y = []
            classNumber:dict = {}
            
            hash_dict: dict = {}
            

            for beat_type, signals in dataDict.items():
                
                
                
                if len(signals) < threshold:
                    APP_LOGGER.warning(f"Tipo di battito {beat_type.value[0]} ha meno di {threshold} segnali, ignorato per la SMOTE")
                    for signal in signals:
                        ignored_x.append(signal)
                        ignored_y.append(beat_type)  # Usa il valore numerico del BeatType
                else:
                    classNumber[beat_type.value[1]] = len(signals)
                    for signal in signals:
                        tensor_hash = hashlib.sha256(signal.numpy().tobytes()).hexdigest()
                        hash_dict[tensor_hash] = True
                        X.append(signal.reshape(-1).numpy())  
                        y.append(beat_type.value[1])          
                  
            X = np.stack(X)  # shape: (n_samples, 560)
            y = np.array(y)  # shape: (n_samples,)            
                        
            sampling_strategy = {
                k: v if (v > 15_000 or k == 0) else (v*200 if v <= 100 else (v*14 if v < 500 else (v*8 if v < 1000 else v*4)))
                for k, v in classNumber.items()
            }
            
            APP_LOGGER.info('-'*100)
            APP_LOGGER.info("Esecuzione della SMOTE...")
            APP_LOGGER.info(sampling_strategy)
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=MITBIHDataset.__RANDOM_SEED, k_neighbors=1)    
            X_res, y_res = smote.fit_resample(X, y)
            APP_LOGGER.info('-'*100)
            
            APP_LOGGER.info("Dati dopo la SMOTE:")
            for beat_type in BeatType:
                if BeatType.isBeat(beat_type):
                    count = np.sum(y_res == beat_type.value[1])
                    APP_LOGGER.info(f"{beat_type.value[0]}: {count} segnali")
            
            X_res = X_res.reshape(-1, 2, 280)  # ripristino della forma originale
            
            
            
            for xi, yi in zip(X_res, y_res):
                beat_type = BeatType.tokenize(int(yi))
                tensor_signal = torch.tensor(xi).view(self._channels.value, 280)
                
                tensor_hash = hashlib.sha256(tensor_signal.numpy().tobytes()).hexdigest()
               
                if len(tensor_signal.shape) == 1:
                    tensor_signal = tensor_signal.unsqueeze(dim=0)
               
                self._windows[window_counter] = {
                    'signal_fragment' : tensor_signal,
                    'beatType': beat_type,
                    'class' : torch.tensor([yi], dtype=torch.long),
                    'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
                    'syntetic': False if hash_dict.get(tensor_hash) != None and hash_dict[tensor_hash] == True else True,
                    #'record_name': record_name,
                }
                window_counter += 1
                
            for tensor_signal, beat_type in zip(ignored_x, ignored_y):
                
                self._windows[window_counter] = {
                    'signal_fragment' : tensor_signal,
                    'beatType': beat_type,
                    'class' : torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
                    'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
                    'syntetic': False,
                    #'record_name': record_name,
                }
                
                window_counter += 1
        else:
            # for beat_type, signals in dataDict.items():
            #     for signal in signals:
            #         self._windows[window_counter] = {
            #             'signal_fragment' : signal,
            #             'beatType': beat_type,
            #             'class' : torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
            #             'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
            #             #'record_name': record_name,
            #         }
            #         window_counter += 1
            temp_list = []
            for beat_type, signals in dataDict.items():
                for signal in signals:
                    temp_list.append({
                        'signal_fragment': signal,
                        'beatType': beat_type,
                        'class': torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
                        'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
                        'syntetic': False,
                        # 'record_name': record_name,
                    })

            # Shuffle dell'intera lista
            random.shuffle(temp_list)
            
            # Ricopia nella struttura self._windows con nuovo ordine casuale
            for window_counter, item in enumerate(temp_list):
                self._windows[window_counter] = item
      
        #APP_LOGGER.info(f"Finestre create: {len(self._windows.keys())}")
        APP_LOGGER.info(f"Finestre create: {window_counter}")
            
          
        # Calcolo dei pesi delle classi
        if self._mode == DatasetMode.TRAINING:
            APP_LOGGER.info("-"*100)
            APP_LOGGER.info("Calcolo dei pesi")
            classes_number: Dict[int, int] = {}
            
            all_classes = [self._windows[i]['class'].item() for i in self._windows]
            all_classes = sorted(all_classes)
            
            for n in all_classes:
                if classes_number.get(n) is None:
                    classes_number[n] = 1
                else:
                    classes_number[n] += 1
                    
            for k, v in classes_number.items():
                p = v/len(all_classes) * 100
                APP_LOGGER.info(f"Classe {k} ({p:.4f}%): {v}")
            
            unique_classes = np.unique(all_classes)
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_classes)
            self._class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            categories_number: Dict[int, int] = {}
            all_categories = [self._windows[i]['category'].item() for i in self._windows]
            all_categories = sorted(all_categories)
            
            for n in all_categories:
                if categories_number.get(n) is None:
                    categories_number[n] = 1
                else:
                    categories_number[n] += 1
                    
            for k, v in categories_number.items():
                p = v/len(all_categories) * 100
                APP_LOGGER.info(f"Categoria {k} ({p:.4f}%): {v}")
            
            unique_caregories = np.unique(all_categories)
            
            
            
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_caregories, y=all_categories)
            self._category_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            APP_LOGGER.info(f"Classi trovate: {unique_classes}")
            APP_LOGGER.info(f"Catregorie trovate: {unique_caregories}")
            APP_LOGGER.info(f"Pesi delle classi calcolati:\n{self._class_weights}")
            APP_LOGGER.info(f"Pesi delle categorie calcolati:\n{self._category_weights}")
    
    
    @classmethod    
    def _formatTime(cls, time: str) -> float:        
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


    def get(self, idx: int) -> dict:
        return self._windows[idx]


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self._dataMode is DatasetDataMode.BEAT_CLASSIFICATION:
            window_data = self._windows[idx]
            return window_data['signal_fragment'], window_data['class'], window_data['category']
        
        elif self._dataMode is DatasetDataMode.BPM_REGRESSION: 
            window_data = self._windows[idx]
            return window_data['signal'], window_data['BPM']

        else:
            raise ValueError(f"Modalità {self._dataMode} non valida") 


    


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

    def plot_record_with_annotations(self, record_name: str, output_filepath: str, seconds_per_row: int = 10):
        """
        Visualizza il segnale di un record con tutte le annotazioni dei battiti e dei tag,
        generando un grafico per ogni sequenza di secondi e unendoli in un unico file.

        Args:
            record_name (str): Il nome del record da visualizzare (es. '100').
            output_filepath (str): Il percorso completo dove salvare l'immagine del grafico finale.
            seconds_per_row (int): Quanti secondi visualizzare su ogni riga del grafico.
        """
        if record_name not in MITBIHDataset._ALL_RECORDS:
            APP_LOGGER.error(f"Record '{record_name}' non trovato nel dataset.")
            return

        signal = MITBIHDataset._ALL_SIGNALS_DICT[record_name]
        beat_annotations = MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_name]
        tag_annotations = MITBIHDataset._ALL_SIGNALS_TAG_ANNOTATIONS_DICT[record_name]
        sample_rate = MITBIHDataset.__SAMPLE_RATE

        if signal is None or sample_rate is None:
            APP_LOGGER.error(f"Dati del segnale o sample rate non disponibili per il record '{record_name}'.")
            return

        samples_per_row = seconds_per_row * sample_rate
        total_samples = signal.shape[1]
        num_rows = int(np.ceil(total_samples / samples_per_row))

        row_images = []
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            APP_LOGGER.info(f"Creata directory per il salvataggio del grafico: {output_dir}")

        APP_LOGGER.info(f"Generazione del grafico per il record '{record_name}' ({num_rows} righe)...")
        for i in tqdm(range(num_rows), desc=f"Plotting record {record_name}"):
            start_sample = i * samples_per_row
            end_sample = min((i + 1) * samples_per_row, total_samples)
            
            current_segment_signal = signal[:, start_sample:end_sample]
            x_values = np.arange(start_sample, end_sample)

            plt.figure(figsize=(max(12, 2*seconds_per_row), 4))
            plt.plot(x_values, current_segment_signal[0].numpy(), label='Segnale ECG MLII')
            if current_segment_signal.shape[0] > 1: # Plot V1 if available
                plt.plot(x_values, current_segment_signal[1].numpy(), label='Segnale ECG V1')
            
            plt.title(f"Record {record_name} - Sezione {i+1}/{num_rows} (Campioni {start_sample}-{end_sample})")
            plt.xlabel("Campioni")
            plt.ylabel("Ampiezza normalizzata")
            plt.legend()
            plt.grid()
            plt.ylim(0, 1) # Assicurati che l'asse Y sia normalizzato

            # Aggiungi annotazioni dei battiti
            for ann in beat_annotations:
                sample_pos = ann["sample_pos"]
                beat_type = ann["annotation"]
                if start_sample <= sample_pos < end_sample:
                    plt.axvline(x=sample_pos, color='red', linestyle='--', linewidth=0.8)
                    plt.text(sample_pos, 0.95, str(beat_type.value), color='red', rotation=90, fontsize=8, ha='center', va='top')

            # Aggiungi annotazioni dei tag
            for ann in tag_annotations:
                sample_pos = ann["sample_pos"]
                tag_type = ann["annotation"]
                if start_sample <= sample_pos < end_sample:
                    plt.axvline(x=sample_pos, color='blue', linestyle=':', linewidth=0.8)
                    plt.text(sample_pos, 0.90, str(tag_type.value), color='blue', rotation=0, fontsize=8, ha='center', va='bottom')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            row_images.append(Image.open(buf).convert("RGB"))

        if not row_images:
            APP_LOGGER.info(f"Nessuna immagine generata per il record {record_name}.")
            return

        # Concatena tutte le immagini delle righe
        widths, heights = zip(*(img.size for img in row_images))
        max_width = max(widths)
        total_height = sum(heights)

        concatenated_img = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in row_images:
            concatenated_img.paste(img, (0, y_offset))
            y_offset += img.size[1]

        concatenated_img.save(output_filepath)
        APP_LOGGER.info(f"Grafico completo del record '{record_name}' salvato in: {output_filepath}")
      
       
    def plot_all_windows_for_record(self, record_name: str, output_dir:str):
        """
        Plotta tutte le finestre di un record in modo continuo.
        """
        if record_name not in self._record_windows:
            APP_LOGGER.info(f"Record {record_name} non trovato in self._file_windows.")
            return

        windows = self._record_windows[record_name]
        num_windows = len(windows)
        if num_windows == 0:
            APP_LOGGER.info(f"Nessuna finestra trovata per il record {record_name}.")
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
            APP_LOGGER.info(f"Nessuna immagine generata per le finestre di {record_name}.")
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
        APP_LOGGER.info(f"Immagine concatenata salvata in: {output_filepath}")
        
        
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
            APP_LOGGER.info(f"Nessuna finestra caricata per la modalità {self._mode.name}.")
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
            APP_LOGGER.info("Nessun valore di BPM trovato.")
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
            APP_LOGGER.info(f"Creata directory: {output_dir}")

        # Salva il grafico
        plt.savefig(output_filepath)
        plt.close(fig) # Chiudi la figura per liberare memoria


        APP_LOGGER.info(f"Diagramma della distribuzione dei BPM con elenco file salvato in: {output_filepath}")

    def plot_class_distribution(self, save_path: str | None = None, show_plot: bool = False):
        """
        Realizza e salva/visualizza un diagramma della distribuzione delle classi
        per la modalità "beat classification".

        Args:
            save_path (str, optional): Il percorso completo dove salvare l'immagine del grafico.
                                       Se None, il grafico non viene salvato.
            show_plot (bool): Se True, il grafico viene visualizzato.
        """
        if self._dataMode != DatasetDataMode.BEAT_CLASSIFICATION:
            APP_LOGGER.warning("La distribuzione delle classi è rilevante solo per la modalità 'beat classification'.")
            return

        if not self._windows:
            APP_LOGGER.info("Nessuna finestra caricata per la modalità 'beat classification'.")
            return

        all_classes = [self._windows[i]['class'].item() for i in self._windows]
        
        # Conta le occorrenze di ogni classe
        class_counts = pd.Series(all_classes).value_counts().sort_index()

        # Mappa i numeri delle classi ai loro nomi leggibili
        class_labels = {
            0: "Normal (N, L, R, B, |)",
            1: "SVEB (A, a, J, j, S)",
            2: "VEB (V, E, e)",
            3: "Fusion (F, f)",
            4: "Ventricular Flutter Wave (!)",
            5: "Paced Beat (/)",
            6: "Unknown Beat (Q)"
        }
        
        # Prepara i dati per il plot
        labels = [class_labels.get(c, f"Classe {c}") for c in class_counts.index]
        counts = class_counts.values

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel("Classe di Battito")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione delle Classi di Battito")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                APP_LOGGER.info(f"Creata directory per il salvataggio del grafico: {output_dir}")
            plt.savefig(save_path)
            APP_LOGGER.info(f"Grafico della distribuzione delle classi salvato in: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
