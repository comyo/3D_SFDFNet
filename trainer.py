import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from config import DEVICE

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Performs one epoch of training."""
    model.train() # Enables Dropout and BatchNorm
    total_loss = 0
    
    for x_voxel, x_tabular, y_label in loader:
        x_voxel, x_tabular, y_label = x_voxel.to(DEVICE), x_tabular.to(DEVICE), y_label.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(x_voxel, x_tabular)
        loss = criterion(out, y_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates the model and returns R² score."""
    model.eval() # IMPORTANT: Disables Dropout
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_voxel, x_tabular, y_label in loader:
            x_voxel, x_tabular, y_label = x_voxel.to(DEVICE), x_tabular.to(DEVICE), y_label.to(DEVICE)
            out = model(x_voxel, x_tabular)
            
            all_preds.append(out.cpu())
            all_targets.append(y_label.cpu())
    
    preds = torch.cat(all_preds, dim=0).squeeze().numpy()
    targets = torch.cat(all_targets, dim=0).squeeze().numpy()
    
    # Handle single sample edge case
    if preds.ndim == 0:
        preds = np.array([preds])
        targets = np.array([targets])

    return r2_score(targets, preds)