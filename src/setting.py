from typing import Final
import os

APP_FOLDER: Final[str] = os.path.sep.join(os.getcwd().split(os.path.sep)[0:-1])

NUM_LAYERS: Final[int] = 4
D_MODEL: Final[int] = 32
NUM_HEADS: Final[int] = 8
DFF: Final[int] = D_MODEL*4
DROPOUT_RATE: Final[float] = 0.1

# Percorsi e configurazioni
DATASET_PATH: Final[str] = os.path.join(APP_FOLDER, 'Dataset', 'mitbih_database')
OUTPUT_PATH: Final[str] = os.path.join(APP_FOLDER, 'output')
NUM_EPOCHS: Final[int] = 10
LEARNING_RATE: Final[float] = 0.001

# Configurazioni del segnale
WINDOW_SIZE: Final[int] = 10
WINDOW_STRIDE: Final[int] = 5