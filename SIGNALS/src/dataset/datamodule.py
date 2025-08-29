from enum import Enum, auto
from matplotlib import pyplot as plt
from PIL import Image
import torch
import io
import os

from tqdm import tqdm

from dataset.dataset import MITBIHDataset, DatasetMode, SplitMode
from torch.utils.data import DataLoader




class Mitbih_datamodule:
    
    def __init__(self, 
            datasetFolder:str,  
            batch_size: int = 1, 
            num_workers: int  = 1,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            prefetch_factor: int = 2,
            training_transform_pipe = None,
            validation_transform_pipe = None,
            splitMode: SplitMode = SplitMode.RANDOM,
            random_seed: int = 120,
            use_smote_on_training:bool = False,
            use_smote_on_validation: bool = False
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
        # self._sample_per_window = sample_per_window
        # self._sample_per_stride = sample_per_stride
        # self._sample_rate = sample_rate

   
        
        self._prefetch_factor: int = prefetch_factor
        self._persistent_workers: bool = persistent_workers
        self._pin_memory: bool = pin_memory
        
        if self._num_workers == 0:
            self._persistent_workers = False
            self._pin_memory = False
            self._prefetch_factor = 1
        
            
        MITBIHDataset.initDataset(
            path=datasetFolder, 
            random_seed=random_seed, 
            windowing_offset=400,
            splitMode=splitMode
            # use_smote_on_training=use_smote_on_training,
            # use_smote_on_validation=use_smote_on_validation
        )
     
        self._TRAINING_DATASET = MITBIHDataset(mode=DatasetMode.TRAINING, transform_pipe= training_transform_pipe)
        self._VALIDATION_DATASET = MITBIHDataset(mode=DatasetMode.VALIDATION, transform_pipe= validation_transform_pipe)
        self._TEST_DATASET = MITBIHDataset(mode=DatasetMode.TEST, transform_pipe= validation_transform_pipe)
    

      
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAINING_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VALIDATION_DATASET, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST_DATASET, batch_size=1, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def get_train_dataset(self) :
        return self._TRAINING_DATASET
    
    def get_val_dataset(self) :
        return self._VALIDATION_DATASET
    
    def get_test_dataset(self) :
        return self._TEST_DATASET