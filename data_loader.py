import pandas as pd
import numpy as np
import os
import torch
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Tuple, Optional, Any

# --- Custom Dataset ---
class HybridDataset(Dataset):
    """A custom PyTorch Dataset that loads pre-processed voxel and tabular tensors."""
    def __init__(self, df, preprocessed_folder, preprocessed_tabular_tensors):
        self.df = df
        self.preprocessed_folder = preprocessed_folder
        self.preprocessed_tabular_tensors = preprocessed_tabular_tensors
        self.target_column = 'OMPs_Rejection' # Hardcoded target for safety
        
        # Helper function to sanitize OMPs_ID for file naming
        def sanitize_filename(name):
            sanitized = re.sub(r'[\u200b-\u200f\ufeff]+', '', str(name)).strip()
            sanitized = sanitized.replace('ﬂ', 'fl').replace('ﬃ', 'ffi')
            return re.sub(r'[^a-zA-Z0-9_ -]', '', sanitized).replace(' ', '_')
        
        self.id_names = df['OMPs_ID'].apply(sanitize_filename).tolist()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        voxel_filename = f"{self.id_names[idx]}.pt"
        voxel_path = os.path.join(self.preprocessed_folder, voxel_filename)
        
        # Load voxel tensor
        # FIX: Added weights_only=True to suppress PyTorch FutureWarning
        voxel_tensor = torch.load(voxel_path, weights_only=True)
        
        # Get tabular tensor
        tabular_tensor = self.preprocessed_tabular_tensors[idx]
        
        # Get Label
        y_label = torch.tensor([self.df.iloc[idx][self.target_column]], dtype=torch.float32)
        
        return voxel_tensor, tabular_tensor, y_label

# --- Data Processing Functions ---
def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df.columns = [col.strip().replace(' ', '_').replace('\xa0', '_').replace('(', '').replace(')', '') for col in df.columns]
    return df

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Helper function to process data (log transform, numeric conversion, and missing values)."""
    
    # Full feature list
    tabular_features_list = [
        'Number', 'OMPs_MW', 'Min_projection', 'Max_projection', 'Molecular_radius',
        'Stokes_Radius_SASA', 'pKa1', 'pKa2', 'log_Kow', 'Molecular_charge', 'log_D',
        'Molar_volume', 'Density', 'Diffusion_coefficient', 'log_D_pH5.5', 'log_D_pH7.4',
        'WS', 'MinPartial_Charge', 'MaxPartial_Charge', 'distribution_coefficient',
        'Eccentricity', 'Dipole_approx', 'MinCharge', 'MaxCharge', 'RotatableBonds',
        'TPSA', 'AvgBondLength', 'MaxBondLength', 'MinBondLength', 'AvgBondAngle',
        'MinBondAngle', 'LogP', 'MWCO', 'Zeta_potential', 'Contact_angle',
        'Pure_water_permeability__bar', 'pore_radius', 'RMS_roughness', 'S_SASA',
        'Charge_product', 'Pure_water_flux', 'Pressure', 'pH', 'Temperature',
        'Filtration_duration', 'OMPs_concentration', 'Cross_flow_velocity'
    ]

    # 1. Force conversion to numeric (coerce errors to NaN)
    for col in tabular_features_list:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Log Transformations
    log_transform_features = ['Diffusion_coefficient', 'WS', 'MWCO',
                              'Pressure', 'Filtration_duration', 'OMPs_concentration']
                                
    for col in log_transform_features:
        if col in df.columns and col in tabular_features_list:
            # Ensure data > 0 before log to prevent errors
            df.loc[:, col] = df[col].apply(lambda x: x if pd.notnull(x) and x > 0 else 1e-6)
            df.loc[:, col] = np.log1p(df[col])
    
    # 3. Handle Missing Columns/Values
    missing_cols = [col for col in tabular_features_list if col not in df.columns]
    if missing_cols:
        tabular_features_list = [col for col in tabular_features_list if col not in missing_cols]
    
    for col in tabular_features_list:
        if df[col].isnull().any():
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)
    
    return df, tabular_features_list

def preprocess_tabular(df: pd.DataFrame, scaler: StandardScaler, xgb_model: Any, 
                       tabular_features_list: List[str], 
                       one_hot_encoder: Optional[OneHotEncoder]=None) -> Tuple[List[torch.Tensor], OneHotEncoder, int]:
    """Transforms tabular data using scaler, XGBoost leaves, and OneHotEncoder."""
    
    # Scale data
    processed_data = scaler.transform(df[tabular_features_list])
    
    # Get leaf indices from XGBoost
    leaf_indices = xgb_model.apply(processed_data)
    
    # One-Hot Encoding
    if one_hot_encoder is None:
        # Fit on training data
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one_hot_features = one_hot_encoder.fit_transform(leaf_indices)
    else:
        # Transform using existing encoder
        one_hot_features = one_hot_encoder.transform(leaf_indices)
        
    # Ensure output is a dense NumPy array
    if hasattr(one_hot_features, 'toarray'):
        one_hot_features = one_hot_features.toarray()

    # Convert to list of tensors
    tabular_tensors = [torch.tensor(f.flatten(), dtype=torch.float32) for f in one_hot_features]
    
    return tabular_tensors, one_hot_encoder, one_hot_features.shape[1]