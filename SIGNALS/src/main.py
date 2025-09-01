import argparse
from enum import Enum, auto
import math
import os
from typing import Final, List, Tuple
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
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
from trainer import Trainer

from dataset.datamodule import Mitbih_datamodule
from dataset.dataset import ArrhythmiaType, SplitMode


def check_pytorch_cuda() -> bool:
    
    
    setting.APP_LOGGER.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.version.cuda is None:
        setting.APP_LOGGER.error(f"Cuda Version: {torch.version.cuda}")
        setting.APP_LOGGER.error("CUDA is not available on this system.")
        setting.APP_LOGGER.error(f"Available GPUs: {torch.cuda.device_count()}")
        return False
    
    else:
        setting.APP_LOGGER.info(f"Cuda Version: {torch.version.cuda}")
        setting.APP_LOGGER.info("CUDA is available on this system.")
        setting.APP_LOGGER.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            setting.APP_LOGGER.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        return True
    

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
    
    check_pytorch_cuda()
    
    from nn_models.ECG_CNN import ECG_CNN_2D
    from nn_models.resnet import ResNet18,ResNet34
    from nn_models.visionTransformer import ViT1,ViT2
    from nn_models.segFormer import SegFormer
    from trainer import Schedulers
    
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    modes = [ SplitMode.LAST_RECORD]#SplitMode.RANDOM,
    
    
    start_lr:float = 0.005
    num_epochs = 100
    seeds:list = []
    
    models = [
        ECG_CNN_2D(),
        ResNet18(),
        ResNet34(),
        ViT1(num_classes=5),
        ViT2(num_classes=5)
    ]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=5)
    

    for mode in modes:
        
        match mode:
            case SplitMode.RANDOM:
                seeds = [48**i for i in range(1,4)]
            case _ :
                seeds = [48]
            
        for seed in seeds:

            datamodule = Mitbih_datamodule(
                datasetFolder=r"C:\Users\Utente\Desktop\Progetto_DataMining\Dataset\mitbih_database", 
                batch_size=150, 
                num_workers=5,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                training_transform_pipe = train_transforms,
                validation_transform_pipe = None,
                random_seed=seed,
                use_smote_on_validation=True,
                splitMode=mode
            )
            
            # print(f"training size: {len(datamodule.get_train_dataset())}")
            # print(f"validation size: {len(datamodule.get_val_dataset())}")
            # print(f"Test size: {len(datamodule.get_test_dataset())}")
        
        

           
            
            for model in models:
                
                optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs, eta_min=2.5e-5
                )
                
                trainer = Trainer(
                    workingDirectory=os.path.join(setting.OUTPUT_PATH, mode.value, f"seed_{seed}"),
                    datamodule=datamodule,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                setting.APP_LOGGER.info(f"TRAINING {model.__class__.__name__}")
                
                trainer.train(
                    num_epochs=100,
                    lr=0.0005,
                
                )
                
                try:
                    
                    setting.APP_LOGGER.info(f"TESTING {model.__class__.__name__}")
                    
                    trainer.evaluate_model(
                        #checkpoint_path="C:\\Users\\Utente\\Desktop\\Progetto_DataMining\\SIGNALS\\Data\\Models\\seed_120\\ViT1\\checkpoints\\checkpoint_epoch_95_val_loss_0.0436.pt",
                        ignore_index=-100,
                        unique_labels=[0, 1, 2, 3, 4],
                        label_mapper= lambda idx: ArrhythmiaType.toEnum(idx).__str__()
                        
                    )
                    
                except Exception as e:
                    print(e)
                
                
            
        
    
    
if __name__ == "__main__":
    main()