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
    


class ArrhythmiaType(Enum):
    NOR = (0, [100, 105, 215], "NOR")
    LBB = (1, [109, 111, 214], "LBB")
    RBB = (2, [118, 124, 212], "RBB")
    PVC = (3, [106, 223], "PVC")
    PAC = (4, [207, 209, 232], "PAC")
    
    def __str__(self) -> str:
        return self.value[2]
    
    @lru_cache(maxsize=1)
    @classmethod
    def toList(cls) -> List[Tuple[int, List[int], str]]:
        return list(map(lambda c: c.value, cls))
  
    @lru_cache(maxsize=5)
    @classmethod
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
    _ALL_SIGNALS_DICT: Dict[int, torch.Tensor] = {}
    _WINDOWS: Dict[int, Dict[str, Any]] = {}
    
    # Dizionario per memorizzare le finestre di ogni record
    _TYPE_RECORD_WINDOWS: Dict[int, Dict[int, torch.Tensor]] = {}

    _TRAIN_DATASET: Dict[str, Dict[int, list]] = {}
    _TEST_DATASET: Dict[str, Dict[int, list]] = {}

    @classmethod
    def __resetDataset(cls) -> None:
        cls._DATASET_PATH = ""
        cls._FILES_CHEKED = False
        cls.__USE_SMOTE = False
        cls._ALL_SIGNALS_DICT.clear()
        cls._TYPE_RECORD_WINDOWS.clear()
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
            use_smote:bool = False
        ):
        """
        Imposta il percorso del dataset e carica staticamente tutti i dati.
        Questo metodo dovrebbe essere chiamato una volta all'inizio del programma.
        """
        
        cls.__resetDataset()
        
        if cls._FILES_CHEKED and cls._DATASET_PATH == path:
            APP_LOGGER.info(f"Il percorso del dataset è già impostato su: {cls._DATASET_PATH}")
            return
        
        cls.__RANDOM_SEED = random_seed
        cls._DATASET_PATH = path
        cls.__USE_SMOTE = use_smote
        
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
        APP_LOGGER.info("caricamneto dei dati ")
        if not cls.__load_signals():
            raise RuntimeError("Errore durante il caricamento dei segnali")
            
        APP_LOGGER.info("Completato")
        
        
        #========================================================================#
        # Finestratura dei dati
        #========================================================================#
        APP_LOGGER.info("Realizzazione delle finestre")
        if not cls.__makeWindows(offset=windowing_offset, window_size=window_size):
            raise RuntimeError("Errore durante la realizzazione delle finestre")
        APP_LOGGER.info("Completato")
        
        
        #========================================================================#
        # Divisione dei dati in training, validation e test
        #========================================================================#
        APP_LOGGER.info("Divisione dei dati in training e validation")
        
        cls._TRAIN_DATASET, cls._TEST_DATASET = cls.__makeDatasets()
        
        
        #========================================================================#
        # SMOTE
        #========================================================================#
        
        if use_smote:
            pass
        
        #========================================================================#
        # STFT
        #========================================================================#
        SFTF_hop_length = max(0.0, min(1.0, SFTF_hop_length))
        cls.__makeSpectrogram(window_size=SFTF_window_size, hop_length=SFTF_hop_length)
        
        
        
    @classmethod
    def __load_signals(cls) -> bool:
    
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
                    
                    for idx, col in enumerate(df.columns):
                        df = df.rename(columns={df.columns[idx]: df.columns[idx].replace('\'', '')})

                    # -- leggo la colonna MLII --
                    signal: torch.Tensor = torch.from_numpy(df[MITBIHDataset._MLII_COL].values).unsqueeze(0)
                    
                    #-- normalizzo il segnale --
                    signal = (signal - MITBIHDataset._MIN_VALUE) / (MITBIHDataset._MAX_VALUE - MITBIHDataset._MIN_VALUE) 
        
                    # -- Salva il segnale per il record corrente --
                    cls._ALL_SIGNALS_DICT[record_name] = signal.float() 
                    #========================================================================#
                    
                progress_bar.update(1)
            return True
        
        except Exception as e:
            APP_LOGGER.error(f"Errore durante il caricamento dei segnali: {e}")
            return False
        
        finally:
            progress_bar.close()
            
    @classmethod 
    def __makeWindows(cls, offset: int, window_size: int = 3600) -> bool:
        
        progress_bar = tqdm(total=len(ArrhythmiaType.toList()), desc="")
        
        try:    
            for (atype, records, label) in ArrhythmiaType.toList():
                
                atype_records_windows: dict = {}
                
                for record_name in records:
                    progress_bar.set_description(f"{label} - record {record_name}")
                    
                    atype_records_windows[record_name] = cls._ALL_SIGNALS_DICT[record_name][offset:(648000+offset):window_size]
                
                cls._TYPE_RECORD_WINDOWS[atype] = atype_records_windows
                progress_bar.update(1)
            return True
        
        except Exception as e:
            APP_LOGGER.error(f"Errore durante il caricamento dei segnali: {e}")
            return False
        
        finally:
            progress_bar.close()
        
        
    @classmethod
    def __makeDatasets(cls, train_ratio: float = 150/180, test_ratio:float = 30/180) -> Tuple[Dict[str, Dict[int, list]], Dict[str, Dict[int, list]]]:
        
        train_dict_windows: Dict[str, Dict[int, list]] = {}
        test_dict_windows:  Dict[str, Dict[int, list]] = {}
        
        #per ogni tipologia di aritmia
        for (atype, _, label) in ArrhythmiaType.toList():
            windows: list = []
            
            #ottengo i record di quella tipologia
            records_dict = cls._TYPE_RECORD_WINDOWS[atype]
            
            #per ogni record prendo le sue finestre e realizzo
            #una lista unica di finestre per la tipologia
            for record in records_dict.keys():
                windows.extend(records_dict[record])
                
            #divido le finestre in train e test
            train_indices, test_indices = train_test_split(
                windows,
                train_size=train_ratio,
                test_size=test_ratio,
                random_state=MITBIHDataset.__RANDOM_SEED
            )
            
            data_dict1: Dict[int, list] = {}
            data_dict2: Dict[int, list] = {}
            
            for i, j in enumerate(train_indices):
                data_dict1[i] = [
                    windows[j],
                    None,
                    False
                ]
                
            for i, j in enumerate(test_indices):
                data_dict2[i] = [
                    windows[j],
                    None,
                    False
                ]
                
            train_dict_windows[label] = data_dict1
            test_dict_windows[label]  = data_dict2
            
        return (train_dict_windows, test_dict_windows)
    
   
    @classmethod
    def __makeSpectrogram(cls, window_size: int, hop_length: float, signal_len: int = 3600) -> None:
        
        window = np.hanning(window_size)
        hop_size = int(window_size * hop_length)
        num_frames = 1 + (3600 - window_size) // hop_size
        
        
        for dataset in [cls._TRAIN_DATASET, cls._TEST_DATASET]:
            for _, windows_dict in dataset.items():
                for (key_idx, data) in windows_dict.items():
                    signal_tensor = data[0]
                    signal_np = signal_tensor.numpy().squeeze()
                    
                    stft_matrix = np.zeros((window_size // 2 + 1, num_frames), dtype=np.complex64)

                    for i in range(num_frames):
                        start = i * hop_size
                        end = start + window_size
                        frame = signal_np[start:end] * window
                        spectrum = np.fft.rfft(frame)  # FFT solo fino a Nyquist
                        stft_matrix[:-1, i] = spectrum

                    # Magnitudo dello spettrogramma
                    spectrogram = np.abs(stft_matrix)
                    spectrogram_db = 20 * np.log10(spectrogram + 1e-8)
               
        
                    # Resize a 256x256
                    spectrogram_resized = resize(spectrogram_db, (256, 256), mode='reflect', anti_aliasing=True)
        
                    data[1] = torch.tensor(spectrogram_resized, dtype=torch.float32).unsqueeze(0)
        
    def __new__(cls, *args, **kwargs):
        return super(MITBIHDataset, cls).__new__(cls)  
        
    
    def __init__(self, mode: DatasetMode):
        assert MITBIHDataset._DATASET_PATH, "Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo."
        assert MITBIHDataset._FILES_CHEKED, "files del dataset non verificati"
        assert len(MITBIHDataset._ALL_SIGNALS_DICT) > 0, "I segnali non sono stati caricati. Usa 'initDataset' per caricarli."
   
        self._mode: DatasetMode = mode    
        self.weights: torch.Tensor | None = None  
  
     
               
        match self._mode:
            case DatasetMode.TRAINING:
                self.__load_data_for_Beat_classification(train_indices)
    
           
            case DatasetMode.VALIDATION | DatasetMode.TEST:
                self.__load_data_for_Beat_classification(test_indices)
    
            case _ :
                raise ValueError(f"Modalità non valida: {self._mode}") 
            
    def getClassWeights(self) -> torch.Tensor:
        assert self._dataMode == DatasetDataMode.BEAT_CLASSIFICATION, "I pesi sono disponibili solo per la classificazione"
        return self._class_weights.detach().clone()
    
    def getCategoryWeights(self) -> torch.Tensor:
        assert self._dataMode == DatasetDataMode.BEAT_CLASSIFICATION, "I pesi sono disponibili solo per la classificazione"
        return self._category_weights.detach().clone()
           
    

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


    
