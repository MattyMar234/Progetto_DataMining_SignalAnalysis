from logging import Logger
from typing import Final
from pathlib import Path
import os

#============================ PATHS ============================#
APP_FOLDER: Final[str] = str(Path(__file__).parent.absolute().parent.absolute())
DATASET_PATH: Final[str] = os.path.join(APP_FOLDER, 'Dataset', 'mitbih_database')

DATA_FOLDER_PATH: Final[str] = os.path.join(APP_FOLDER, "Data")
OUTPUT_PATH: Final[str] = os.path.join(DATA_FOLDER_PATH, 'Models')

if not os.path.exists(DATA_FOLDER_PATH):
    os.makedirs(DATA_FOLDER_PATH)
    
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#============================= Model =============================#
NUM_LAYERS: Final[int] = 4
D_MODEL: Final[int] = 512*2
NUM_HEADS: Final[int] = 8
DFF: Final[int] = D_MODEL*4
DROPOUT_RATE: Final[float] = 0.1

RANDOM_SEED: Final[int] = 42

# Percorsi e configurazioni
#============================= training =============================#
NUM_EPOCHS: Final[int] = 10
LEARNING_RATE: Final[float] = 0.001

AVG_VALIDATION_LOSS_LABEL_NAME: Final[str] = 'avg_loss'

#============================= dataset =============================#
# Configurazioni del segnale
WINDOW_SIZE: Final[int] = 10
WINDOW_STRIDE: Final[int] = 5


#============================= logs =============================#
LOGS_FOLDER: Final[str] = os.path.join(DATA_FOLDER_PATH, 'logs')
LOGGERS_CONFIG_FILE: Final[str] = os.path.join(DATA_FOLDER_PATH, 'loggerConfig.json')

if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

APP_LOGGER: Logger | None = None

APP_LOGGER_NAME: Final[str] = 'appInfo'
CONSOLE_LOGGER_NAME: Final[str] = 'console'