from collections import Counter
import hashlib
import random
from sklearn.model_selection import train_test_split
import torch
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
from torchvision import transforms
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

from setting import APP_LOGGER
from functools import lru_cache
from scipy.signal import stft
from skimage.transform import resize

# Definizione delle modalità del dataset
class DatasetMode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    
class SplitMode(Enum):
    RANDOM = "mode_random"
    LAST_RECORD = "mode_record"
    

class ArrhythmiaType(Enum):
    NOR = (0, [100, 105, 215], "NOR")
    LBB = (1, [109, 111, 214], "LBB")
    RBB = (2, [118, 124, 212], "RBB")
    PVC = (3, [106, 223], "PVC")
    PAC = (4, [207, 209, 232], "PAC")
    
    def __str__(self) -> str:
        return self.value[2]
    
    @classmethod
    def toList(cls) -> List[Tuple[int, List[int], str]]:
        return list(map(lambda c: c.value, cls))
  
    @classmethod
    @lru_cache(maxsize=5)
    def toEnum(cls, value: int) -> 'ArrhythmiaType':
        for member in cls:
            if member.value[0] == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")
  
  
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
    
    _COLS = [_MLII_COL, _V1_COL, _V2_COL, _V3_COL, _V4_COL, _V5_COL]
    
    # lista dei record da utilizzare
    _ALL_RECORDS: Final[list] = _RECORDS_MLII_V1 #+ _RECORDS_MLII_V2 + _RECORDS_MLII_V4 + _RECORDS_MLII_V5# + _RECORDS_V5_V2
    
    _FILES_CHEKED: bool = False
    _DATASET_PATH: str = ""
    _RISOLUZIONE_ADC: int = 11
    _MIN_VALUE: int = 0
    _MAX_VALUE: int = 2**_RISOLUZIONE_ADC - 1
    _MAX_SAMPLE_NUM: int = 650000
    _Fs: Final[int] = 360  # Frequenza di campionamento in Hz

    
    __SAMPLE_RATE: int | None = None
    __RANDOM_SEED: int | None = None
    __USE_SMOTE: bool = True
    
    # Dizionario per memorizzare il segnale di ogni record
 
    # _WINDOWS: Dict[int, Dict[str, Any]] = {}
    
    _TRAIN_DATASET: Dict[str, Dict[int, dict]] = {}
    _TEST_DATASET: Dict[str, Dict[int, dict]] = {}

    @classmethod
    def __resetDataset(cls) -> None:
        cls._DATASET_PATH = ""
        cls._FILES_CHEKED = False
        cls.__USE_SMOTE = False
        cls._TRAIN_DATASET.clear()
        cls._TEST_DATASET.clear()
        

    @classmethod
    def initDataset(
            cls, 
            path: str,  
            *, 
            window_size:int = 3600,
            windowing_offset: int = 0,
            random_seed: int = 42, 
            SFTF_window_size: int = 512,
            SFTF_hop_length: float = 0.50,
            use_smote_on_training:bool = False,
            use_smote_on_validation: bool = False,
            splitMode: SplitMode = SplitMode.RANDOM,
        ):
        """
        Imposta il percorso del dataset e carica staticamente tutti i dati.
        Questo metodo dovrebbe essere chiamato una volta all'inizio del programma.
        """
        
        cls.__resetDataset()
        
        if cls._FILES_CHEKED and cls._DATASET_PATH == path:
            APP_LOGGER.info(f"Il percorso del dataset è già impostato su: {cls._DATASET_PATH}")
            return
        
        cls.__splitMode = splitMode
        cls.__RANDOM_SEED = random_seed
        cls._DATASET_PATH = path
        cls.__USE_SMOTE_TRAINING = use_smote_on_training
        cls.__USE_SMOTE_VALIDATION = use_smote_on_validation
        
        #========================================================================#
        # VERIFICO I FILES
        #========================================================================#
        APP_LOGGER.info("Verifica esitenza directory dataset...")
        
        if not os.path.isdir(path):
            raise FileNotFoundError(f"La directory specificata non esiste: {path}")
        APP_LOGGER.info(f"Directory trovata: {path}")
        
        APP_LOGGER.info("Verifica dei files...")
        for record_name in cls._ALL_RECORDS:
            csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
            txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")
            assert os.path.exists(csv_filepath), f"file {csv_filepath} non trovato" 
            assert os.path.exists(txt_filepath), f"file {txt_filepath} non trovato" 
        APP_LOGGER.info("Tutti i file trovati")
        
        cls._FILES_CHEKED = True
        
        #========================================================================#
        # CARICO I DATI
        #========================================================================#
        APP_LOGGER.info("-"*100)
        APP_LOGGER.info("caricamneto dei dati ")
        signals_dict = cls.__load_signals() 
        
        
        # for k, v in signals_dict.items():
        #     print(f"{k}: {v.shape}")
    
        
        #========================================================================#
        # Finestratura dei dati
        #========================================================================#
        APP_LOGGER.info("-"*100)
        APP_LOGGER.info("Realizzazione delle finestre...")
        type_Record_Dict = cls.__makeWindows(signals_dict=signals_dict, offset=windowing_offset, window_size=window_size)
        
  
        # for k, v in type_Record_Dict.items():
        #     print(f"{k}:")
        #     for k2, l in v.items():
        #         print(f"\t{k2}:")
        #         for i, t in enumerate(l):
        #             print(f"\t\t{i}: {t.shape}")
             
        
        #========================================================================#
        # Divisione dei dati in training, validation e test
        #========================================================================#
        APP_LOGGER.info("-"*100)
        APP_LOGGER.info("Divisione dei dati in training e validation")
        
        cls._TRAIN_DATASET, cls._TEST_DATASET = cls.__makeDatasets(type_Record_Dict=type_Record_Dict)
        
        APP_LOGGER.info("Finestre per il training:")
        for k, v in cls._TRAIN_DATASET.items():
            APP_LOGGER.info(f"- {k}: {len(v)} finestre")
          
        APP_LOGGER.info("Finestre per il test:")
        for k, v in cls._TEST_DATASET.items():
            APP_LOGGER.info(f"- {k}: {len(v)} finestre")
        
        #========================================================================#
        # SMOTE
        #========================================================================#
        
        if cls.__USE_SMOTE_TRAINING:
            APP_LOGGER.info("-"*100)
            APP_LOGGER.info("Applicazione della SMOTE sul dataset di training...")
            cls._TRAIN_DATASET = cls.__apply_smote(cls._TRAIN_DATASET)
            APP_LOGGER.info("SMOTE applicata. Nuova distribuzione del training set:")
            
            total_smote_samples = 0
            for k, v in cls._TRAIN_DATASET.items():
                smote_samples = sum(1 for data_dict in v.values() if data_dict['s'])
                total_smote_samples += smote_samples
                APP_LOGGER.info(f"- {k}: {len(v)} finestre (di cui {smote_samples} sintetiche)")
            APP_LOGGER.info(f"Totale campioni sintetici aggiunti: {total_smote_samples}")
        
        if cls.__USE_SMOTE_VALIDATION:
            APP_LOGGER.info("-"*100)
            APP_LOGGER.info("Applicazione della SMOTE sul dataset di validazione...")
            cls._TEST_DATASET = cls.__apply_smote(cls._TEST_DATASET)
            APP_LOGGER.info("SMOTE applicata. Nuova distribuzione del validazione set:")
            
            total_smote_samples = 0
            for k, v in cls._TEST_DATASET.items():
                smote_samples = sum(1 for data_dict in v.values() if data_dict['s'])
                total_smote_samples += smote_samples
                APP_LOGGER.info(f"- {k}: {len(v)} finestre (di cui {smote_samples} sintetiche)")
            APP_LOGGER.info(f"Totale campioni sintetici aggiunti: {total_smote_samples}")
              
        
        #========================================================================#
        # STFT
        #========================================================================#
        APP_LOGGER.info("-"*100)
        APP_LOGGER.info("Realizzazione degli spettrogrammi...")
        SFTF_hop_length = max(0.0, min(1.0, SFTF_hop_length))
        cls.__makeSpectrogram(window_size=SFTF_window_size, hop_length=SFTF_hop_length)
        
        
        
    @classmethod
    def __load_signals(cls) -> Dict[int, torch.Tensor]:
    
        ALL_SIGNALS_DICT: Dict[int, torch.Tensor] = {}
        progress_bar = tqdm(total=len(ArrhythmiaType.toList()), desc="Caricamento Records")
        
        try:    
            for (_, records, label) in ArrhythmiaType.toList():
                for record_name in records:
                    progress_bar.set_description(f"{label} - record {record_name}")
                
                    
                    csv_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}.csv")
                    #txt_filepath = os.path.join(MITBIHDataset._DATASET_PATH, f"{record_name}annotations.txt")
            
                    #========================================================================#
                    # ESTRAGGO IL SEGNALE
                    #========================================================================#
                    # --- Leggi il segnale dal file CSV ---
                    df = pd.read_csv(csv_filepath)
                    
                    channels_name = []
                    
                    for idx, col in enumerate(df.columns):
                        rm = df.columns[idx].replace('\'', '')
                        df = df.rename(columns={df.columns[idx]: rm})
                        
                        if rm in MITBIHDataset._COLS:
                            channels_name.append(rm)
                    
                    signals: List[torch.Tensor] = []    
                    
                    for channel in channels_name:
                        #print(channel)
                        s: torch.Tensor = torch.from_numpy(df[channel].values)
                        s = (s - MITBIHDataset._MIN_VALUE) / (MITBIHDataset._MAX_VALUE - MITBIHDataset._MIN_VALUE) 
                        signals.append(s)
            
                    signal = torch.stack(signals, dim=0)
            
                    # -- Salva il segnale per il record corrente --
                    ALL_SIGNALS_DICT[record_name] = signal.float() 
                    #========================================================================#
                    
                progress_bar.update(1)
            return ALL_SIGNALS_DICT
        finally:
            progress_bar.close()
            
    @classmethod 
    def __makeWindows(cls, signals_dict: Dict[int, torch.Tensor], offset: int, window_size: int = 3600) -> Dict[int, Dict[int, torch.Tensor]]:
        
        TYPE_RECORD_WINDOWS: Dict[int, Dict[int, torch.Tensor]] = {}
        progress_bar = tqdm(total=len(ArrhythmiaType.toList()), desc="")
        offset = max(0, min(2000, offset))
        
        try:    
            for (atype, records, label) in ArrhythmiaType.toList():
                
                atype_records_windows: dict = {}
                
                for record_name in records:
                    progress_bar.set_description(f"{label} - record {record_name}")
                    
                    signal = signals_dict[record_name]
                    x = signal[:, offset:(MITBIHDataset._MAX_SAMPLE_NUM-2000+offset)]
                     # split lungo la dimensione temporale → lista di [N, window_size]
                    chunks = torch.split(x, window_size, dim=1)
                    
                    # rimodello: da lista di [N, window_size] → [num_chunks, N, window_size]
                    stacked = torch.stack(chunks, dim=0)
                    
                    windows_list = windows_list = [w for w in stacked.reshape(-1, window_size)]
                    atype_records_windows[record_name] = windows_list
                    
                    # print(f"signal: {signal.shape}")
                    # print(f"chunks: {chunks[0].shape}")
                    # print(f"stacked: {stacked.shape}")
                    # print(f"windows_list: {len(windows_list)}")
                    # print(f"type: {type(windows_list)}")
                    # os._exit(0)
                
                TYPE_RECORD_WINDOWS[atype] = atype_records_windows
                progress_bar.update(1)
            
            return TYPE_RECORD_WINDOWS
        finally:
            progress_bar.close()
        
        
    @classmethod
    def __makeDatasets(cls, type_Record_Dict: Dict[int, Dict[int, torch.Tensor]], train_ratio: float = 150/180, test_ratio:float = 30/180) -> Tuple[Dict[str, Dict[int, dict]], Dict[str, Dict[int, dict]]]:
        
        train_dict_windows: Dict[str, Dict[int, dict]] = {}
        test_dict_windows:  Dict[str, Dict[int, dict]] = {}
        
        #per ogni tipologia di aritmia
        for (atype, record_list, label) in ArrhythmiaType.toList():
            windows: list = []
            train_tensors, test_tensors = [], []
            
            #ottengo i record di quella tipologia
            records_dict = type_Record_Dict[atype]
            
               
            #divido le finestre in train e test
            match cls.__splitMode:
                case SplitMode.RANDOM:
                    
                    #per ogni record prendo le sue finestre e realizzo
                    #una lista unica di finestre per la tipologia
                    for record in records_dict.keys():
                        windows.extend(records_dict[record])
                    
                    train_tensors, test_tensors = train_test_split(
                        windows,
                        train_size=train_ratio,
                        test_size=test_ratio,
                        random_state=MITBIHDataset.__RANDOM_SEED
                    )
                
                case SplitMode.LAST_RECORD:
                    train_records = record_list[0:-1]
                    test_record = record_list[-1]
                    
                    for record in train_records:
                        train_tensors.extend(records_dict[record])
                    
                    test_tensors.extend(records_dict[test_record])
                
                
                case _ :
                    raise Exception("Invalid dataset splitMode")
            
                       
            data_dict1: Dict[int, dict] = {}
            data_dict2: Dict[int, dict] = {}
            
            for dict_d, tensors_list in zip([data_dict1, data_dict2], [train_tensors, test_tensors]):
                for i, T in enumerate(tensors_list):
                    dict_d[i] = {"x1": T, "x2":None, "s":False, "y":torch.tensor(atype)}
                  
            train_dict_windows[label] = data_dict1
            test_dict_windows[label]  = data_dict2
            
        return (train_dict_windows, test_dict_windows)
    
    @classmethod
    def __apply_smote(cls, dataset: Dict[str, Dict[int, dict]]):
        
        import hashlib
        
        """
        Applica SMOTE per bilanciare il dataset di training.
        
        Args:
            dataset (Dict[str, Dict[int, dict]]): Il dizionario del dataset di training.
            
        Returns:
            Dict[str, Dict[int, dict]]: Il dizionario del dataset aggiornato con i campioni SMOTE.
        """
        
        # 1. Combina tutti i dati e le etichette in due liste
        all_data = []
        all_labels = []
        original_data_dict = {}
        counter = 0

        for label, windows_dict in dataset.items():
            for key_idx, data_dict in windows_dict.items():
                # Aggiungi l'elemento al mapping
                all_data.append(data_dict['x1'].numpy().flatten())
                all_labels.append(data_dict['y'])
                
                # Mappa l'indice originale ai dati, inclusi i metadati
                original_data_dict[counter] = data_dict
                counter += 1

        APP_LOGGER.info(f"Dimensione dataset prima di SMOTE: {len(all_data)}")
        APP_LOGGER.info(f"Distribuzione classi prima di SMOTE: {Counter(all_labels)}")
        
        # 2. Applica SMOTE
        smote = SMOTE(random_state=cls.__RANDOM_SEED, sampling_strategy={0: 200, 1: 200, 2: 200, 3: 200, 4: 200})
        X_resampled, y_resampled = smote.fit_resample(all_data, all_labels)
        
        APP_LOGGER.info(f"Dimensione dataset dopo SMOTE: {len(X_resampled)}")
        APP_LOGGER.info(f"Distribuzione classi dopo SMOTE: {Counter(y_resampled)}")
        
        # 3. Ricrea il dizionario del dataset con i dati sintetici
        new_dataset = {str(arr_type): {} for arr_type in ArrhythmiaType.toList()}
        
        original_size = len(all_data)
        synthetic_idx_start = original_size
        
        # Ricostruisci il dizionario per i dati originali
        for i in range(original_size):
            original_data_dict[i]['x1'] = torch.from_numpy(np.array(X_resampled[i])).float()
            
            # Recupera il tipo di aritmia corretto
            arr_type_label = ArrhythmiaType.toEnum(y_resampled[i]).value[2]
            
            # Aggiungi l'elemento al nuovo dizionario
            new_dataset[arr_type_label][i] = original_data_dict[i]
            
        # Ricostruisci il dizionario per i dati sintetici
        for i in range(synthetic_idx_start, len(X_resampled)):
            
            synt_data = torch.from_numpy(X_resampled[i]).float()
            synt_label_enum = ArrhythmiaType.toEnum(y_resampled[i])
            synt_label_str = synt_label_enum.value[2]
            
            new_data_dict = {
                "x1": synt_data,
                "x2": None, # Lo spettrogramma verrà calcolato dopo
                "s": True, # Segna come sintetico
                "y": synt_label_enum
            }
            
            # Aggiungi al nuovo dizionario
            new_dataset[synt_label_str][i] = new_data_dict
            
        return new_dataset
   
    @classmethod
    def __makeSpectrogram(cls, window_size: int, hop_length: float, signal_len: int = 3600) -> None:
        
        window = np.hanning(window_size)
        hop_size = int(window_size * hop_length)
        num_frames = 1 + (3600 - window_size) // hop_size
        
        # Barra esterna per i dataset
        dataset_bar = tqdm(total=2, desc="Dataset", position=0)

        # Barra type
        type_bar = tqdm(total=5, desc="Type", position=1)

        # Barra per le finestre (0-100 percento)
        frame_bar = tqdm(total=100, desc="Frames %", position=2)
        
        try:
            for (dataset, dataset_name) in zip([cls._TRAIN_DATASET, cls._TEST_DATASET], ["Train", "Test"]):
                dataset_bar.update(1)
                str_ = f"{dataset_name} Dataset"
                dataset_bar.set_description(f"{str_:<17}")
                dataset_bar.refresh() 
                
                type_bar.n = 0
                type_bar.refresh()
                for atype, windows_dict in dataset.items():
                    str_ = f"Type: {atype}"
                    type_bar.set_description(f"{str_:<17}")
                    
                    elements = len(windows_dict.items())
                    frame_bar.n = 0
                    frame_bar.refresh()
                
                    for (idx,(key_idx, data)) in enumerate(windows_dict.items()):
                        str_ = f"window: {idx+1}" 
                        frame_bar.set_description(f"{str_:<17}")
                        frame_bar.refresh()

                        signal_tensor = data["x1"]
                        signal_np = signal_tensor.numpy().squeeze()
                        
                        stft_matrix = np.zeros((window_size // 2, num_frames), dtype=np.complex64)

                        for i in range(num_frames):
                            start = i * hop_size
                            end = start + window_size
                            frame = signal_np[start:end] * window
                            
                            # FFT solo fino a Nyquist
                            #ho 512/2+1 = 257 valori. Ogni passo corrisponde a 360/512 = 0.703125 Hz
                            spectrum = np.fft.rfft(frame)  
                            stft_matrix[:, i] = spectrum[0:-1]

                        # Magnitudo dello spettrogramma. abs => modulo complesso
                        spectrogram = np.abs(stft_matrix)**2
                        spectrogram_db = np.log10(spectrogram + 1e-8)
                
            
                        # Resize a 256x256
                        spectrogram_resized = resize(spectrogram_db, (256, 256), mode='reflect', anti_aliasing=True)
            
                         # ======================================================================== #
                        # NORMALIZZAZIONE tra 0 e 1
                        # ======================================================================== #
                        min_val = np.min(spectrogram_resized)
                        max_val = np.max(spectrogram_resized)
                        
                        # Previene la divisione per zero se max_val e min_val sono uguali
                        if max_val > min_val:
                            spectrogram_normalized = (spectrogram_resized - min_val) / (max_val - min_val)
                        else:
                            # Se tutti i valori sono uguali, lo spettrogramma normalizzato sarà tutto a 0
                            spectrogram_normalized = np.zeros_like(spectrogram_resized)

                        # Salva lo spettrogramma normalizzato
                        data["x2"] = torch.tensor(spectrogram_normalized, dtype=torch.float32).unsqueeze(0)
                        
            
                        #data["x2"] = torch.tensor(spectrogram_resized, dtype=torch.float32).unsqueeze(0)
            
                        # update percentuale
                        perc = int((idx + 1) / elements * 100)
                        frame_bar.n = perc
                        frame_bar.refresh()
                        
                    type_bar.update(1)
                    type_bar.refresh() 
        finally:
            dataset_bar.close()
            type_bar.close()
            frame_bar.close()         
        
        
        
    def __new__(cls, *args, **kwargs):
        return super(MITBIHDataset, cls).__new__(cls)  
    
    def __len__(self) -> int:
        return len(self.__index_mapping)

    def __getitem__(self, idx: int) -> dict:
        assert 0 <= idx < len(self.__index_mapping), f"Indice fuori range. Deve essere compreso tra 0 e {len(self.__index_mapping) - 1}."
        
        data_cloned = self.get(idx)
        
        if self._transforms is not None:
            x = data_cloned['x2']
            data_cloned['x2'] = self._transforms(x)
            
        return data_cloned
    
    def get(self, idx: int) -> dict:
        return {
            'x1': self.__index_mapping[idx]['x1'].detach().clone(),  # Segnale ECG
            'x2': self.__index_mapping[idx]['x2'].detach().clone(),  # Spettrogramma
            'y': self.__index_mapping[idx]['y'].detach().clone(),    # Etichetta
            's': self.__index_mapping[idx]['s']  # Indica se il campione è sintetico
        }
    
    def __init__(self, mode: DatasetMode, transform_pipe: transforms.Compose | None = None):
        assert MITBIHDataset._DATASET_PATH, "Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo."
        assert MITBIHDataset._FILES_CHEKED, "files del dataset non verificati"
        #assert len(MITBIHDataset._ALL_SIGNALS_DICT) > 0, "I segnali non sono stati caricati. Usa 'initDataset' per caricarli."
        assert transform_pipe is None or isinstance(transform_pipe, transforms.Compose), "'transform_pipe' deve essere di tipo 'transforms.Compose' o None."
   
        self._mode: DatasetMode = mode    
        self.__weights: torch.Tensor | None = None
        self._transforms: transforms.Compose | None = transform_pipe  
        
     
               
        match self._mode:
            case DatasetMode.TRAINING:
                self.__data = MITBIHDataset._TRAIN_DATASET
                
            case DatasetMode.VALIDATION | DatasetMode.TEST:
                self.__data = MITBIHDataset._TEST_DATASET
    
            case _ :
                raise ValueError(f"Modalità non valida: {self._mode}") 
          
        self.__index_mapping: List[dict] = []
        
        for _, windows_dict in self.__data.items():
            for _, w_dict in windows_dict.items():
                self.__index_mapping.append(w_dict)
                
        # Calcola i pesi delle classi solo se in modalità TRAINING
        if self._mode == DatasetMode.TRAINING:
            self.__calculate_class_weights()
            APP_LOGGER.info(f"Pesi delle classi calcolati: {self.__weights}")
        
        
    def __calculate_class_weights(self):
        """Calcola e imposta i pesi delle classi per il set di training."""
        labels = []
        for d in self.__index_mapping:
            labels.append(d['y'].item())
            
        # Trova tutte le classi uniche presenti nel set di dati
        classes = np.unique(labels)
        
        # Calcola i pesi inversamente proporzionali alla frequenza delle classi
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        self.__weights = torch.tensor(weights, dtype=torch.float32)
            
    def get_class_weights(self) -> torch.Tensor | None:
        """Restituisce i pesi delle classi."""
        return self.__weights
           
    def show_sample_spectrogram(self, index: int):
        """
        Mostra il segnale del dataset e il relativo spettrogramma per un dato indice.
        I due grafici sono visualizzati in un'unica finestra.
        """
        if index >= len(self.__index_mapping) or index < 0:
            raise IndexError(f"Indice fuori range. L'indice deve essere compreso tra 0 e {len(self.__index_mapping) - 1}.")
        
        # Estrai i dati per l'indice specificato
        data_point = self.__index_mapping[index]
        signal = data_point['x1']
        spectrogram = data_point['x2']
        label = data_point['y']
        is_synthetic = data_point['s']
        
        # Verifica che sia il segnale che lo spettrogramma siano disponibili
        if signal is None or spectrogram is None:
            raise ValueError("I dati del segnale o dello spettrogramma non sono disponibili per questo indice. Assicurati che il dataset sia stato inizializzato correttamente.")
            
        # Converti i dati da PyTorch a NumPy per il plotting
        signal_np = signal.squeeze().numpy()
        spectrogram_np = spectrogram.squeeze().numpy()
        
        # Crea una figura con due sottotracciati
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Titolo generale
        synthetic_text = " (sintetico)" if is_synthetic else ""
        fig.suptitle(f"Sample {index} - Categoria: {ArrhythmiaType.toEnum(label)}{synthetic_text}", fontsize=16)
        
        # Plot del segnale
        ax1.plot(signal_np, color="red")
        ax1.set_title("Segnale ECG")
        ax1.set_xlabel("Campioni")
        ax1.set_ylabel("Ampiezza Normalizzata")
        ax1.grid(True)
        
        # Plot dello spettrogramma
        im = ax2.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='inferno')#jet
        ax2.set_title("Spettrogramma")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Frequenza (bin)")
        
        # Aggiungi una colorbar per lo spettrogramma
        fig.colorbar(im, ax=ax2, label="Intensità (dB)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Regola il layout per il titolo
        plt.show()

    # def __load_data_for_Beat_classification(self, indices: List[int]):
    #     indices = sorted(indices)
        
    #     idx: int = 0
    #     counter: int = 0
    #     window_counter: int = 0
    #     done: bool = False
        
    #     progress_bar = tqdm(total=len(indices))
        
    #     dataDict: Dict[BeatType, List[torch.Tensor]] = {}
        
    #     try:
    #         for record_name in MITBIHDataset._ALL_RECORDS:
    #             progress_bar.set_description(f"record {record_name}")
                
    #             if done:  break
                
    #             annotations = MITBIHDataset._ALL_SIGNALS_BEAT_ANNOTATIONS_DICT[record_name]
    #             signal = MITBIHDataset._ALL_SIGNALS_DICT[record_name]
                
                
    #             for annotation in annotations:
                    
    #                 #se ho preso tutti gli elementi
    #                 if idx >= len(indices):
    #                     done=True
    #                     break
                    
    #                 #se non ho ancora raggiugnto l'indice dell'elemento da utilizzare
    #                 elif indices[idx] > counter:
    #                     counter += 1
    #                     continue
                    
    #                 #se h oraggiugnto l'indice dell'elemeneto da utilizzare
    #                 elif indices[idx] == counter:
    #                     progress_bar.update(1)
    #                     counter += 1
    #                     idx+=1
                        
                    
                    
    #                 beat_type = annotation["annotation"]
    #                 sample_pos = annotation["sample_pos"]
    #                 #time = annotation["time"]
                    
    #                 #Verifico il tipo di annotazione
    #                 if not BeatType.isBeat(beat_type):
    #                     continue
                    
    #                 start = sample_pos - int(self._samples_per_window / 2)
    #                 end = sample_pos + int(self._samples_per_window / 2)


    #                 #ignoro il campione se non ci sta nella finestra
    #                 # if start < 0 or end >= MITBIHDataset._MAX_SAMPLE_NUM:
    #                 #     continue
                    
    #                 if start < 0:
    #                     offset = abs(start)
    #                     start = 0
    #                     end = end + offset
                    
    #                 if end > MITBIHDataset._MAX_SAMPLE_NUM - 1:
    #                     offset = end - (MITBIHDataset._MAX_SAMPLE_NUM -1)
    #                     end = MITBIHDataset._MAX_SAMPLE_NUM -1
    #                     start = start - offset
                      
    #                 # # Extract and pad signal fragment
    #                 signal_fragment = signal[:, start:end]
                    
    #                 if dataDict.get(beat_type) is None:
    #                     dataDict[beat_type] = [signal_fragment]
    #                 else:
    #                     dataDict[beat_type].append(signal_fragment)
                    
    #     finally:
    #         progress_bar.close()  
            
    #     APP_LOGGER.info("Valori di ogni classe:")
    #     for beat_type, signals in dataDict.items():
    #         APP_LOGGER.info(f"{beat_type.value[0]}: {len(signals)} segnali")
        
    #     if self._mode == DatasetMode.TRAINING and MITBIHDataset.__USE_SMOTE:
    #         APP_LOGGER.info(f"Applicazione della SMOTE")   
            
            
    #         X = []
    #         y = []
            
    #         threshold = 2
    #         ignored_x = []
    #         ignored_y = []
    #         classNumber:dict = {}
            
    #         hash_dict: dict = {}
            

    #         for beat_type, signals in dataDict.items():
                
                
                
    #             if len(signals) < threshold:
    #                 APP_LOGGER.warning(f"Tipo di battito {beat_type.value[0]} ha meno di {threshold} segnali, ignorato per la SMOTE")
    #                 for signal in signals:
    #                     ignored_x.append(signal)
    #                     ignored_y.append(beat_type)  # Usa il valore numerico del BeatType
    #             else:
    #                 classNumber[beat_type.value[1]] = len(signals)
    #                 for signal in signals:
    #                     tensor_hash = hashlib.sha256(signal.numpy().tobytes()).hexdigest()
    #                     hash_dict[tensor_hash] = True
    #                     X.append(signal.reshape(-1).numpy())  
    #                     y.append(beat_type.value[1])          
                  
    #         X = np.stack(X)  # shape: (n_samples, 560)
    #         y = np.array(y)  # shape: (n_samples,)            
                        
    #         sampling_strategy = {
    #             k: v if (v > 15_000 or k == 0) else (v*200 if v <= 100 else (v*14 if v < 500 else (v*8 if v < 1000 else v*4)))
    #             for k, v in classNumber.items()
    #         }
            
    #         APP_LOGGER.info('-'*100)
    #         APP_LOGGER.info("Esecuzione della SMOTE...")
    #         APP_LOGGER.info(sampling_strategy)
    #         smote = SMOTE(sampling_strategy=sampling_strategy, random_state=MITBIHDataset.__RANDOM_SEED, k_neighbors=1)    
    #         X_res, y_res = smote.fit_resample(X, y)
    #         APP_LOGGER.info('-'*100)
            
    #         APP_LOGGER.info("Dati dopo la SMOTE:")
    #         for beat_type in BeatType:
    #             if BeatType.isBeat(beat_type):
    #                 count = np.sum(y_res == beat_type.value[1])
    #                 APP_LOGGER.info(f"{beat_type.value[0]}: {count} segnali")
            
    #         X_res = X_res.reshape(-1, 2, 280)  # ripristino della forma originale
            
            
            
    #         for xi, yi in zip(X_res, y_res):
    #             beat_type = BeatType.tokenize(int(yi))
    #             tensor_signal = torch.tensor(xi).view(self._channels.value, 280)
                
    #             tensor_hash = hashlib.sha256(tensor_signal.numpy().tobytes()).hexdigest()
               
    #             if len(tensor_signal.shape) == 1:
    #                 tensor_signal = tensor_signal.unsqueeze(dim=0)
               
    #             self._windows[window_counter] = {
    #                 'signal_fragment' : tensor_signal,
    #                 'beatType': beat_type,
    #                 'class' : torch.tensor([yi], dtype=torch.long),
    #                 'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
    #                 'syntetic': False if hash_dict.get(tensor_hash) != None and hash_dict[tensor_hash] == True else True,
    #                 #'record_name': record_name,
    #             }
    #             window_counter += 1
                
    #         for tensor_signal, beat_type in zip(ignored_x, ignored_y):
                
    #             self._windows[window_counter] = {
    #                 'signal_fragment' : tensor_signal,
    #                 'beatType': beat_type,
    #                 'class' : torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
    #                 'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
    #                 'syntetic': False,
    #                 #'record_name': record_name,
    #             }
                
    #             window_counter += 1
    #     else:
    #         # for beat_type, signals in dataDict.items():
    #         #     for signal in signals:
    #         #         self._windows[window_counter] = {
    #         #             'signal_fragment' : signal,
    #         #             'beatType': beat_type,
    #         #             'class' : torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
    #         #             'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
    #         #             #'record_name': record_name,
    #         #         }
    #         #         window_counter += 1
    #         temp_list = []
    #         for beat_type, signals in dataDict.items():
    #             for signal in signals:
    #                 temp_list.append({
    #                     'signal_fragment': signal,
    #                     'beatType': beat_type,
    #                     'class': torch.tensor([BeatType.getBeatClass(beat_type)], dtype=torch.long),
    #                     'category': torch.tensor([BeatType.getBeatCategory(beat_type)], dtype=torch.long),
    #                     'syntetic': False,
    #                     # 'record_name': record_name,
    #                 })

    #         # Shuffle dell'intera lista
    #         random.shuffle(temp_list)
            
    #         # Ricopia nella struttura self._windows con nuovo ordine casuale
    #         for window_counter, item in enumerate(temp_list):
    #             self._windows[window_counter] = item
      
    #     #APP_LOGGER.info(f"Finestre create: {len(self._windows.keys())}")
    #     APP_LOGGER.info(f"Finestre create: {window_counter}")
            
          
    #     # Calcolo dei pesi delle classi
    #     if self._mode == DatasetMode.TRAINING:
    #         APP_LOGGER.info("-"*100)
    #         APP_LOGGER.info("Calcolo dei pesi")
    #         classes_number: Dict[int, int] = {}
            
    #         all_classes = [self._windows[i]['class'].item() for i in self._windows]
    #         all_classes = sorted(all_classes)
            
    #         for n in all_classes:
    #             if classes_number.get(n) is None:
    #                 classes_number[n] = 1
    #             else:
    #                 classes_number[n] += 1
                    
    #         for k, v in classes_number.items():
    #             p = v/len(all_classes) * 100
    #             APP_LOGGER.info(f"Classe {k} ({p:.4f}%): {v}")
            
    #         unique_classes = np.unique(all_classes)
    #         class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_classes)
    #         self._class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
    #         categories_number: Dict[int, int] = {}
    #         all_categories = [self._windows[i]['category'].item() for i in self._windows]
    #         all_categories = sorted(all_categories)
            
    #         for n in all_categories:
    #             if categories_number.get(n) is None:
    #                 categories_number[n] = 1
    #             else:
    #                 categories_number[n] += 1
                    
    #         for k, v in categories_number.items():
    #             p = v/len(all_categories) * 100
    #             APP_LOGGER.info(f"Categoria {k} ({p:.4f}%): {v}")
            
    #         unique_caregories = np.unique(all_categories)
            
            
            
    #         class_weights = compute_class_weight(class_weight='balanced', classes=unique_caregories, y=all_categories)
    #         self._category_weights = torch.tensor(class_weights, dtype=torch.float32)
            
    #         APP_LOGGER.info(f"Classi trovate: {unique_classes}")
    #         APP_LOGGER.info(f"Catregorie trovate: {unique_caregories}")
    #         APP_LOGGER.info(f"Pesi delle classi calcolati:\n{self._class_weights}")
    #         APP_LOGGER.info(f"Pesi delle categorie calcolati:\n{self._category_weights}")
    
    



    
