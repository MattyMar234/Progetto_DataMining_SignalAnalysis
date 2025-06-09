from matplotlib import pyplot as plt
import pytorch_lightning as pl
from PIL import Image
import torch
import io
import os

from tqdm import tqdm

from dataset.dataset import DatasetChannels, MITBIHDataset, DatasetMode, DatasetDataMode
from torch.utils.data import DataLoader


class Mitbih_datamodule(pl.LightningDataModule):
    
    def __init__(self, 
            datasetFolder:str, 
            datasetDataMode: DatasetDataMode, 
            datasetChannels: DatasetChannels,
            batch_size: int = 1, 
            num_workers: int  = 1,
            sample_rate: int | None = None,
            sample_per_window: int | None = None,
            sample_per_stride: int | None = None,
            pin_memory: bool = True,
            persistent_workers: bool = True
        ):
        super().__init__()
        
        assert os.path.exists(datasetFolder), f"La cartella {datasetFolder} non esiste"
        assert batch_size > 0, "Batch size must be greater than 0"
        assert num_workers >= 0, "Number of workers must be greater than or equal to 0"
        assert isinstance(datasetFolder, str), "Dataset folder must be a string"
        assert len(datasetFolder) > 0, "Dataset folder must not be empty"
    
        self._datasetFolder = datasetFolder
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sample_per_window = sample_per_window
        self._sample_per_stride = sample_per_stride
        self._sample_rate = sample_rate
        self._datasetDataMode = datasetDataMode
        self._datasetChannels = datasetChannels
        
        self._prefetch_factor: int | None = 1
        self._persistent_workers: bool = persistent_workers
        self._pin_memory: bool = pin_memory
        
        if self._num_workers == 0:
            self._persistent_workers = False
            self._pin_memory = False
            self._prefetch_factor = None
            
        MITBIHDataset.initDataset(path=datasetFolder, sample_rate=sample_rate, random_seed=120)
     
        self._TRAINING_DATASET = MITBIHDataset(dataMode=self._datasetDataMode, channels=self._datasetChannels, mode=DatasetMode.TRAINING, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
        self._VALIDATION_DATASET = MITBIHDataset(dataMode=self._datasetDataMode, channels=self._datasetChannels, mode=DatasetMode.VALIDATION, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
        self._TEST_DATASET = MITBIHDataset(dataMode=self._datasetDataMode, channels=self._datasetChannels, mode=DatasetMode.TEST, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
    
    def print_all_window(self, columNumber: int, save_path: str, dataset: MITBIHDataset) -> None:
        
        os.makedirs(save_path, exist_ok=True)
        window_imgs = []

        for i in tqdm(range(len(dataset)), desc=f"Plotting windows"):
            img = self.print_window(save_path="", dataset=dataset, index=i, show_plot= False, getFile=True)
            if img is not None:
                window_imgs.append(img.convert("RGB"))

        if not window_imgs:
            print(f"Nessuna immagine generata")
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

        concatenated_img.save(save_path)
        print(f"Immagine concatenata salvata in: {save_path}")
        
    
    
    def print_window(self, save_path: str, dataset: MITBIHDataset, index: int, show_plot: bool = False, getFile: bool = False) -> Image.Image | None:
        assert isinstance(dataset, MITBIHDataset)
        #assert os.path.exists(save_path)
        
        window_data = dataset.get(index)
        
        if self._datasetDataMode == DatasetDataMode.BEAT_CLASSIFICATION:
            signal = window_data['signal_fragment']
            label = window_data['beatType']
            label_str = f"Beat Type: {label}"
        
        time = torch.arange(signal.shape[1]) / self._sample_rate
        
        plt.figure(figsize=(12, 6))
        plt.title(label_str)
        plt.plot(time.numpy(), signal[0].numpy())
        plt.plot(time.numpy(), signal[1].numpy())
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            if getFile:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img = Image.open(buf)
                return img
            else:
                # Assicurati che la directory esista
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"Plot salvato in: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close() # Chiudi la figura se non deve essere mostrata
    
    def print_all_training_ecg_signals(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        
        for s in sorted(MITBIHDataset.TRAINING_RECORDS):
            self._TRAINING_DATASET.plot_all_windows_for_record(s, output_folder)
            #MITBIHDataset.plot_all_windows_for_record()
    def print_training_record(self, record_name: str, output_folder: str) -> None:
        self._TRAINING_DATASET.plot_all_windows_for_record(record_name, output_folder)
            
    def print_validation_plot_bpm_distribution(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        self._VALIDATION_DATASET.plot_bpm_distribution(output_folder)
      
    def print_training_plot_bpm_distribution(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        self._TRAINING_DATASET.plot_bpm_distribution(output_folder)
      
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAINING_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VALIDATION_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def get_train_dataset(self) :
        return self._TRAINING_DATASET
    
    def get_val_dataset(self) :
        return self._VALIDATION_DATASET
    
    def get_test_dataset(self) :
        return self._TEST_DATASET