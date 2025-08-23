import argparse
from enum import Enum, auto
import math
from typing import Final, List, Tuple
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score, recall_score
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, cohen_kappa_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import setting

from dataset.datamodule import Mitbih_datamodule


def check_pytorch_cuda() -> bool:
    #Globals.APP_LOGGER.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        setting.APP_LOGGER.info("CUDA is available on this system.")
        setting.APP_LOGGER.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            setting.APP_LOGGER.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        setting.APP_LOGGER.info("CUDA is not available on this system.")
        return False
    
def main():

    # parser = argparse.ArgumentParser(description="Script per configurare e addestrare un modello Transformer sui dati MIT-BIH.")
    
 
    # # Percorsi e configurazioni
    # parser.add_argument("--dataset_path", type=str, default=setting.DATASET_PATH, help=f"Percorso del dataset (default: {setting.DATASET_PATH})")
    # parser.add_argument("--checkpoint_path", type=str, default=None, help="Percorso di un addestramento precedente (default: None)")
    # parser.add_argument("--output_path", type=str, default=setting.OUTPUT_PATH, help=f"Percorso di output per i risultati (default: {setting.OUTPUT_PATH})")
    # parser.add_argument("--num_epochs", type=int, default=setting.NUM_EPOCHS, help=f"Numero di epoche di training (default: {setting.NUM_EPOCHS})")
    # parser.add_argument("--lr", type=float, default=setting.LEARNING_RATE, help=f"Valore iniziale del learning rate (default: {setting.LEARNING_RATE})")
    # parser.add_argument("--batch_size", type=int, default=16, help=f"Dimensione del batch per il training (default: {16})")
    # parser.add_argument("--num_workers", type=int, default=4, help=f"Numero di worker per il DataLoader (default: {4})")
    
    
    datamodule = Mitbih_datamodule(
        datasetFolder=r"C:\Users\Utente\Desktop\Progetto_DataMining\Dataset\mitbih_database", 
        batch_size=16, 
        num_workers=4,
        sample_rate=360,
        sample_per_window=400,
        sample_per_stride=200,
        pin_memory=True,
        persistent_workers=True
    )
    
    check_pytorch_cuda()
    train_dataset=datamodule.get_train_dataset()
    
    print(len(train_dataset))
    print(train_dataset[0])
    train_dataset.show_sample_spectrogram(400)
    
    
if __name__ == "__main__":
    main()