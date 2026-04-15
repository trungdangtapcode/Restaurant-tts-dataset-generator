import os
from pathlib import Path

class Config:
    # Select GPUs to use. Based on nvidia-smi, 2 A4000s are [0, 1]
    USE_GPUS = [0, 1]  
    INPUT_FILE = "bills-training.json"
    OUTPUT_DIR = Path("dataset_output")
    VARIATIONS_PER_BILL = 3
