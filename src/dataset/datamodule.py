import os
import pytorch_lightning as pl
import torch

from dataset.dataset import MITBIHDataset, DatasetMode
from torch.utils.data import DataLoader


class Mitbih_datamodule(pl.LightningDataModule):
    
    def __init__(self, 
            datasetFolder:str, 
            batch_size: int = 1, 
            num_workers: int  = 1,
            sample_rate: int = 360,
            window_size_t: int = 10,
            window_stride_t: int = 5,
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
        
        self._prefetch_factor: int | None = 1
        self._persistent_workers: bool = persistent_workers
        self._pin_memory: bool = pin_memory
        
        if self._num_workers == 0:
            self._persistent_workers = False
            self._pin_memory = False
            self._prefetch_factor = None
            
        MITBIHDataset.setDatasetPath(datasetFolder)
        MITBIHDataset.init_dataset()
        
        sample_per_window = sample_rate * window_size_t
        sample_per_stride = sample_rate * window_stride_t
            
        self._TRAINING_DATASET = MITBIHDataset(mode=DatasetMode.TRAINING, sample_rate=sample_rate, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
        self._VALIDATION_DATASET = MITBIHDataset(mode=DatasetMode.VALIDATION, sample_rate=sample_rate, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
        self._TEST_DATASET = MITBIHDataset(mode=DatasetMode.TEST, sample_rate=sample_rate, sample_per_window=sample_per_window, sample_per_stride=sample_per_stride)
    
    def print_all_training_ecg_signals(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        
        for s in sorted(MITBIHDataset.TRAINING_RECORDS):
            self._TRAINING_DATASET.plot_all_windows_for_record(s, output_folder)
            #MITBIHDataset.plot_all_windows_for_record()
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAINING_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VALIDATION_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    def train_dataset(self) :
        return self._TRAINING_DATASET