import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE

class VoxelCNN(nn.Module):
    def __init__(self, in_channels, out_features=128):
        super(VoxelCNN, self).__init__()
        # 3D Convolutional layers for voxel data
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(32, out_features, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x

class TabularMLP(nn.Module):
    def __init__(self, in_channels, out_features=128):
        super(TabularMLP, self).__init__()
        # Multi-Layer Perceptron for tabular data
        self.fc1 = nn.Linear(in_channels, out_features) 
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class DeeperFusionModel(nn.Module):
    def __init__(self, voxel_features_flat_size, tabular_features_size, hidden_size_1, hidden_size_2, dropout_rate):
        super(DeeperFusionModel, self).__init__()
        
        # Shared size for initial fusion outputs
        shared_fusion_size = tabular_features_size 
        
        # Projection layers
        self.voxel_fc = nn.Linear(voxel_features_flat_size, shared_fusion_size)
        self.tabular_fc = nn.Linear(tabular_features_size, shared_fusion_size)
        
        # Combined size (Voxel + Tabular + Element-wise Product)
        final_fusion_size = shared_fusion_size * 3 
        
        # Deep fusion layers
        self.fc_fusion1 = nn.Linear(final_fusion_size, hidden_size_1)
        self.fc_fusion2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc_final = nn.Linear(hidden_size_2, 1)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, voxel_features, tabular_features):
        voxel_features_flat = voxel_features.view(voxel_features.size(0), -1)
        
        # Projections
        voxel_fused = self.voxel_fc(voxel_features_flat)
        tabular_fused = self.tabular_fc(tabular_features)
        
        # Fusion: Concatenating features and their element-wise product (cross-feature)
        feature_cross = voxel_fused * tabular_fused
        x_combined = torch.cat([voxel_fused, tabular_fused, feature_cross], dim=1)
        
        # Prediction layers
        x = F.relu(self.fc_fusion1(x_combined))
        x = self.dropout(x)
        x = F.relu(self.fc_fusion2(x))
        x = self.fc_final(x)
        return x

class HybridModel(nn.Module):
    """The complete hybrid model combining CNN for voxels and MLP for tabular data."""
    def __init__(self, tabular_in_channels, num_voxel_channels, hidden_size_1, hidden_size_2, dropout_rate):
        super(HybridModel, self).__init__()
        voxel_out_features = 128
        tabular_out_features = 128
        
        # Feature extractors
        self.voxel_cnn = VoxelCNN(in_channels=num_voxel_channels, out_features=voxel_out_features)
        
        # Temporarily move VoxelCNN to device for dummy forward pass to calculate size
        self.voxel_cnn.to(DEVICE)
        
        # Calculate the flattened size of VoxelCNN output dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_voxel_channels, 32, 32, 32).to(DEVICE)
            dummy_output = self.voxel_cnn(dummy_input)
            self.voxel_flat_size = dummy_output.view(1, -1).size(1)
        
        self.tabular_mlp = TabularMLP(in_channels=tabular_in_channels, out_features=tabular_out_features)
        
        # Fusion and final prediction head
        self.fusion_model = DeeperFusionModel(
            self.voxel_flat_size, tabular_out_features, hidden_size_1, hidden_size_2, dropout_rate
        )

    def forward(self, x_voxel, x_tabular):
        voxel_features = self.voxel_cnn(x_voxel)
        tabular_features = self.tabular_mlp(x_tabular)
        out = self.fusion_model(voxel_features, tabular_features)
        return out