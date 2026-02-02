import torch
import numpy as np
import random
import os

# --- System Settings ---
# Check device availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Random Seed for reproducibility
RANDOM_SEED = 98564

# --- File Paths ---
INPUT_FILE = "structured_data.xlsx"
PREPROCESSED_FOLDER = "preprocessed_voxels"
FOLDS_FILE = 'kfold_splits.joblib'
RESULTS_DIR = 'results'

# --- Data Constants ---
TARGET_COLUMN = 'OMPs_Rejection'
NUM_VOXEL_CHANNELS = 4

# --- Hyperparameters (Fixed from Tuning) ---
TUNING_EPOCHS = 200        # Total training epochs
EVAL_INTERVAL = 1          # Evaluate test R2 every N epochs

# Fixed Best Hyperparameters (from Trial 92/150)
FIXED_LR = 0.000138
FIXED_BATCH_SIZE = 16
FIXED_HIDDEN_SIZE_1 = 512
FIXED_HIDDEN_SIZE_2 = 128
FIXED_DROPOUT = 0.28

# --- Utility Functions ---
def set_seed(seed):
    """Set global random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False