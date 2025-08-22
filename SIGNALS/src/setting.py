import logging
from logging import Logger
from typing import Any, Final
from pathlib import Path
import os
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

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
LEARNING_RATE: Final[float] = 0.0005

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

# APP_LOGGER: Logger | None = None # Will be initialized below



class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages based on their level.
    """
    FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    LOG_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Get the original formatted message without color
        original_message = super().format(record)
        
        # Get the color for the current log level
        color_code = self.LOG_COLORS.get(record.levelno, Fore.WHITE)
        
        # Find the start and end of the levelname in the original message
        # This assumes the format is consistent: "[timestamp] [levelname] message"
        level_name_start = original_message.find(f"[{record.levelname}]")
        
        if level_name_start != -1:
            # Extract parts of the message
            before_level = original_message[:level_name_start + 1] # Include '['
            level_name_part = record.levelname
            after_level = original_message[level_name_start + 1 + len(record.levelname):] # Exclude ']'
            
            # Reconstruct the message with only the levelname colored
            colored_message = f"{before_level}{color_code}{level_name_part}{Style.RESET_ALL}{after_level}"
            return colored_message
        else:
            # Fallback if levelname not found as expected (shouldn't happen with default format)
            return color_code + original_message + Style.RESET_ALL

# Configure the application logger
def setup_logger(name: str, logLevel: int = logging.INFO) -> Logger:
    assert isinstance(name, str), "Logger name must be a string"
    assert isinstance(logLevel, int), "Log level must be an integer"
    
    logger = logging.getLogger(name)
    logger.setLevel(logLevel) # Set the default logging level

    # Prevent adding multiple handlers if the logger is re-initialized
    if not logger.handlers:
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(fmt=ColoredFormatter.FORMAT, datefmt=ColoredFormatter.DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for persistent logs (without color codes)
        file_handler = logging.FileHandler(os.path.join(LOGS_FOLDER, f'{name}.log'))
        file_formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

APP_LOGGER: Logger = setup_logger(name='appInfo', logLevel=logging.INFO)
