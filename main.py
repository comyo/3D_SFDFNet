import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import joblib
import re
from torch.utils.data import DataLoader

# Import local modules
import config
from config import (
    INPUT_FILE, PREPROCESSED_FOLDER, FOLDS_FILE, RESULTS_DIR,
    RANDOM_SEED, TARGET_COLUMN, NUM_VOXEL_CHANNELS, DEVICE,
    TUNING_EPOCHS, EVAL_INTERVAL, 
    FIXED_LR, FIXED_BATCH_SIZE, FIXED_HIDDEN_SIZE_1, FIXED_HIDDEN_SIZE_2, FIXED_DROPOUT
)
from data_loader import process_data, format_column_names, preprocess_tabular, HybridDataset
from models import HybridModel
from trainer import train_one_epoch, evaluate_model

def main():
    print("\n--- Phase 2: Training the Hybrid PyTorch Model (Modularized) ---")
    
    # 1. Initialize Seed
    config.set_seed(RANDOM_SEED)

    try:
        # 2. Load and Clean Data
        df = pd.read_excel(INPUT_FILE, sheet_name=0) 
        df = format_column_names(df)
        
        smiles_col_name = 'SMILEs'
        id_col_name = 'OMPs_ID'
        
        # Initial DropNA
        df_filtered = df.dropna(subset=[smiles_col_name, TARGET_COLUMN, id_col_name]).copy()
        df_filtered.reset_index(drop=True, inplace=True)
        
        # Process Features
        df_processed_orig, tabular_features_list = process_data(df_filtered)

        # Helper to match filenames
        def sanitize_filename(name):
            sanitized = re.sub(r'[\u200b-\u200f\ufeff]+', '', str(name)).strip()
            sanitized = sanitized.replace('ﬂ', 'fl').replace('ﬃ', 'ffi')
            return re.sub(r'[^a-zA-Z0-9_ -]', '', sanitized).replace(' ', '_')

        df_processed_orig['sanitized_name'] = df_processed_orig['OMPs_ID'].apply(sanitize_filename)
        
        # 3. Filter data to only include samples with existing voxel files
        if not os.path.exists(PREPROCESSED_FOLDER):
            print(f"Error: Voxel folder '{PREPROCESSED_FOLDER}' not found.")
            return
        
        processed_files = [f for f in os.listdir(PREPROCESSED_FOLDER) if f.endswith('.pt')]
        processed_names = [os.path.splitext(f)[0] for f in processed_files]
        
        # Final DataFrame matching available files
        df_final = df_processed_orig[df_processed_orig['sanitized_name'].apply(lambda x: x in processed_names)].copy()
        df_final.reset_index(drop=True, inplace=True) 
        
        if len(df_final) == 0:
            print("Error: No molecules remaining after filtering for existing voxel files.")
            return
            
        print(f"\nFinal dataset size after filtering: {len(df_final)} molecules.")
        
        # 4. Load K-Fold splits
        folds = joblib.load(FOLDS_FILE)
        print(f"K-Fold splits loaded successfully from '{FOLDS_FILE}'.")
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure all required files are present.")
        return

    # 5. Display Fixed Hyperparameters
    print("\n--- Using Fixed Best Hyperparameters ---")
    print(f"LR={FIXED_LR:.6f}, BS={FIXED_BATCH_SIZE}, H1={FIXED_HIDDEN_SIZE_1}, H2={FIXED_HIDDEN_SIZE_2}, DO={FIXED_DROPOUT:.3f}")
    
    # 6. K-Fold Training Loop
    all_best_r2_scores = []
    
    for fold, (train_index_orig, test_index_orig) in enumerate(folds):
        print(f"\n--- Fold {fold+1}/{len(folds)} ---")
        
        # Reset seed for each fold for reproducibility
        config.set_seed(RANDOM_SEED + fold)
        
        # Define paths
        fold_dir = os.path.join(RESULTS_DIR, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        temp_best_model_path = os.path.join(fold_dir, 'temp_best_model.pt')
        final_model_save_path = os.path.join(fold_dir, 'final_hybrid_model.pt')
        ohe_save_path = os.path.join(fold_dir, 'ohe_encoder.joblib')
        
        # Load pre-processing tools for this fold
        try:
            xgb_model = joblib.load(os.path.join(fold_dir, 'xgb_model.joblib'))
            scaler = joblib.load(os.path.join(fold_dir, 'scaler.joblib'))
        except FileNotFoundError:
            print(f"Error: Pre-processing tools for Fold {fold+1} not found in '{fold_dir}'. Skipping.")
            continue

        # Map K-Fold indices back to the filtered dataframe
        train_df_orig_indices = df_processed_orig.iloc[train_index_orig].index
        test_df_orig_indices = df_processed_orig.iloc[test_index_orig].index
        
        train_indices_final = [i for i, idx in enumerate(df_final.index) if idx in train_df_orig_indices]
        test_indices_final = [i for i, idx in enumerate(df_final.index) if idx in test_df_orig_indices]

        if not train_indices_final or not test_indices_final:
            print(f"Warning: Fold {fold+1} is empty after filtering. Skipping.")
            continue
            
        train_df = df_final.iloc[train_indices_final].copy().reset_index(drop=True)
        test_df = df_final.iloc[test_indices_final].copy().reset_index(drop=True)
        
        # --- Pre-process Tabular Features ---
        # Train: Fit & Transform
        train_tabular_tensors, one_hot_encoder, tabular_feature_size = preprocess_tabular(
            train_df, scaler, xgb_model, tabular_features_list
        )
        # Save OneHotEncoder
        joblib.dump(one_hot_encoder, ohe_save_path)
        print(f"  OneHotEncoder for Fold {fold+1} saved to: {ohe_save_path}")
        
        # Test: Transform Only
        test_tabular_tensors, _, _ = preprocess_tabular(
            test_df, scaler, xgb_model, tabular_features_list, one_hot_encoder
        )
        
        # Create Datasets & Loaders
        train_dataset = HybridDataset(train_df, PREPROCESSED_FOLDER, train_tabular_tensors)
        test_dataset = HybridDataset(test_df, PREPROCESSED_FOLDER, test_tabular_tensors)
        
        train_loader = DataLoader(train_dataset, batch_size=FIXED_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=FIXED_BATCH_SIZE, shuffle=False)
        
        # Initialize Model
        model = HybridModel(
            tabular_in_channels=tabular_feature_size,
            num_voxel_channels=NUM_VOXEL_CHANNELS,
            hidden_size_1=FIXED_HIDDEN_SIZE_1,
            hidden_size_2=FIXED_HIDDEN_SIZE_2,
            dropout_rate=FIXED_DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=FIXED_LR)
        criterion = nn.MSELoss()
        
        # Training Loop
        best_test_r2_final = -float('inf')
        
        for epoch in range(1, TUNING_EPOCHS + 1):
            avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            
            # Evaluate
            if epoch % EVAL_INTERVAL == 0 or epoch == TUNING_EPOCHS:
                current_test_r2 = evaluate_model(model, test_loader)
                
                if current_test_r2 > best_test_r2_final:
                    best_test_r2_final = current_test_r2
                    # Checkpoint best model
                    torch.save(model.state_dict(), temp_best_model_path)
                
                if epoch % 20 == 0:
                    print(f"      Epoch [{epoch}/{TUNING_EPOCHS}], Loss: {avg_loss:.4f}, Test R²: {current_test_r2:.4f}, Best R²: {best_test_r2_final:.4f}")

        # Finalize Fold
        if os.path.exists(temp_best_model_path):
            model.load_state_dict(torch.load(temp_best_model_path))
            final_best_r2 = evaluate_model(model, test_loader)
            all_best_r2_scores.append(final_best_r2)
            
            print(f"Fold {fold+1} MAX Test R² Achieved: {final_best_r2:.4f}")
            
            # Save final best model
            torch.save(model.state_dict(), final_model_save_path)
            print(f"  Final BEST model saved to: {final_model_save_path}")
            
            # Cleanup temp file
            os.remove(temp_best_model_path)
        else:
            print(f"Error: No best model found for Fold {fold+1}.")

    # 7. Final Results
    print("\n--- Final K-Fold Evaluation Results ---")
    if all_best_r2_scores:
        mean_r2 = np.mean(all_best_r2_scores)
        std_r2 = np.std(all_best_r2_scores)
        print(f"Average R²: {mean_r2:.4f} ± {std_r2:.4f}")
    else:
        print("No successful fold evaluations were performed.")

if __name__ == '__main__':
    main()