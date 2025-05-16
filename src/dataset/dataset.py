from sklearn.model_selection import train_test_split
import torch
import wfdb
import numpy as np
import os
from torch.utils.data import Dataset
from enum import Enum, auto
from typing import Dict, Final, List, Tuple
import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt




# Definizione delle modalità del dataset
class DatasetMode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()
    
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
    
    _RECORDS_V5_V2 = ['104', '102']
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
        for record_name in cls.ALL_RECORDS:
            csv_filepath = os.path.join(path, f"{record_name}.csv")
            txt_filepath = os.path.join(path, f"{record_name}annotations.txt")

            if not os.path.exists(csv_filepath):
                raise FileNotFoundError(f"File CSV non trovato: {csv_filepath}")
            if not os.path.exists(txt_filepath):
                raise FileNotFoundError(f"File TXT non trovato: {txt_filepath}")

        cls._FILE_CHEKED = True
        
        
    @classmethod
    def init_dataset(cls) -> None:
        assert cls._FILE_CHEKED, "files del dataset non verificati"
        
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        train_indices, temp_indices = train_test_split(cls.ALL_RECORDS, test_size=(val_ratio + test_ratio), random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        cls.TRAINING_RECORDS = train_indices
        cls.VALIDATION_RECORDS = val_indices
        cls.TEST_RECORDS = test_indices
        
        # cls._TRAINING_RECORDS = [train_indices[0]]
        # cls._VALIDATION_RECORDS = [val_indices[0]]
        # cls._TEST_RECORDS = [test_indices[0]]
        
    
    def __init__(self, *, mode: DatasetMode, sample_rate: int = 360, sample_per_window: int = 360, sample_per_stride: int = 180):
        assert MITBIHDataset._FILE_CHEKED, "files del dataset non verificati"
        
        if not MITBIHDataset._DATASET_PATH:
            raise ValueError("Il percorso del dataset non è stato impostato. Usa 'setDatasetPath' per impostarlo.")
        

        self.mode = mode
        self.sample_rate = sample_rate
        self.samples_per_window = sample_per_window
        self.samples_per_side = sample_per_stride
        

        if mode == DatasetMode.TRAINING:
            self.record_list = MITBIHDataset.TRAINING_RECORDS
            
        elif mode == DatasetMode.VALIDATION:
            self.record_list = MITBIHDataset.VALIDATION_RECORDS
            
        elif mode == DatasetMode.TEST:
            self.record_list = MITBIHDataset.TEST_RECORDS
            
        else:
            raise ValueError(f"Modalità non valida: {mode}")

        print(f"Caricamento dati per la modalità {self.mode.name} dai record: {self.record_list}")
        
        
    
        self._signals_dict = {} # Dizionario per memorizzare i segnali
        self._windows: Dict[int, Dict[str, any]] = {} 
        self._file_windows: Dict[str, Dict[int, Dict[str, any]]] = {}
        self._load_data()
        
        
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
        if record_name not in self._file_windows:
            print(f"Record {record_name} non trovato in self._file_windows.")
            return

        windows = self._file_windows[record_name]
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
            img = self.plot_windows(idx=i + sum(len(self._file_windows[r]) for r in self.record_list if r < record_name), asfile=False, save=False)
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
        
        
      
        

    def _load_data(self):
        
        windows_counter:int = 0
        current_record_name:str | None = None
        """Carica i dati dai record selezionati (CSV per segnale, TXT per annotazioni)."""
        
        try:
    
            for record_name in self.record_list:
                current_record_name = record_name
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
                
                
                windows_number:float = ((len(signal[0]) - self.samples_per_window) / self.samples_per_side) + 1
                windows_number_int = int(windows_number)
                record_windows: list = []
            
            
                #creazione delle finestre
                for i in range(windows_number_int):
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
                        'tag' : [],
                        'tag_positions' : [],  
                        'beat_time': [],
                    }
                    
                    record_windows.append(window)
                
                # Se ci sono campioni rimanenti, crea una finestra finale
                if windows_number - windows_number_int > 0:
                    start = MITBIHDataset._MAX_SAMPLE_NUM - 1 - self.samples_per_window
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
                        'tag' : [],
                        'tag_positions' : [],  
                        'beat_time': [],
                    }
                    
                    record_windows.append(window)
                
                #print(f"Record {record_name} ha {len(record_windows)} finestre.")
                
                window_pointer: int = 0
                current_window = record_windows[window_pointer]
                window_start = current_window['start']
                window_end = current_window['end']
                i_pointer: int = 0
                j_pointer: int = 0
                
                with open(txt_filepath, 'r') as f:
                    f.readline() # Salta la prima riga (header)
                    
                    for line in f:

                        
                        line = line.strip()  # Rimuovi spazi bianchi
                        parts = line.split() # Dividi la riga in base agli spazi e ignore le stringe vuote

                        # Una riga di annotazione valida dovrebbe avere almeno 3 parti (Time, Sample #, Type)
                        if len(parts) < 3:
                            raise ValueError(f"Riga di annotazione non valida: {line}.")

                        beat_type = SampleType.to_Label(parts[2])
                        sample_pos = int(parts[1])            # Indice del campione
                        time = self._formatTime(parts[0]) # Tempo in secondi
                        
                          
                        match beat_type:
                            case SampleType.NORMAL_N | SampleType.NORMAL_L | SampleType.NORMAL_R | SampleType.NORMAL_B | SampleType.NORMAL_LINE\
                                | SampleType.SVEB_A | SampleType.SVEB_a | SampleType.SVEB_J | SampleType.SVEB_j | SampleType.SVEB_S\
                                | SampleType.VEB_V | SampleType.VEB_E | SampleType.VEB_e \
                                | SampleType.FUSION_F | SampleType.FUSION_f \
                                | SampleType.UNKNOWN_BEAT_Q | SampleType.PACED_BEAT | SampleType.VENTRICULAR_FLUTTER_WAVE:
                                    
                                             
                                #cero tutte le finestre dove posso inserire l'annotazione
                                fist: bool = True
                                inc: bool = False
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
                                #     elif fist:
                                #         inc = True
                                #         fist = False
                                # if inc:
                                #     i_pointer += 1
                                       
                            case SampleType.START_SEG_PLUS | SampleType.COMMENT | SampleType.END_NOISE | SampleType.START_NOISE | SampleType.NOISE | SampleType.CHANGE_IN_SIGNAL_QUALITY:
                                
                                for w in range(0, len(record_windows)):
                                    current_window = record_windows[w]
                                    window_start = current_window['start']
                                    window_end = current_window['end']
                                    
                                    if window_start <= sample_pos <= window_end:
                                        current_window['tag_positions'].append(sample_pos)
                                        current_window['tag'].append(beat_type)
                                     
                            
                            case _:
                                # Se il tipo di battito non è valido, ignora
                                print(f"Tipo di battito sconosciuto: {parts[2]}. Ignorato.")
                                continue

                        
        
                for i in range(len(record_windows)):
                    bpm: float = 0
                    current_window = record_windows[i]
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
                        bpm = 0  # Solo un beat, impossibile calcolare BPM
                    else:
                        bpm = 0  # Nessun beat nella finestra
                    
                    #print(f"Finestra {i} BPM: {current_window['BPM']}")
                    
                    current_window['BPM'] = torch.Tensor([bpm]).float()
                    self._windows[windows_counter] = current_window
                    windows_counter += 1
          
                self._file_windows[record_name] = record_windows
          
        except Exception as e:
            print(f"Errore durante il caricamento dei dati per il record {current_record_name}: {e}")
            raise e
            
           
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


