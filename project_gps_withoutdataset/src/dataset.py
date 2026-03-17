import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BeamDataset(Dataset):
    def __init__(self, scenario_name, mode='train', data_root='./Data/processed'):
        # 路径示例: ./Data/processed/scenario31/X_train.npy
        folder = os.path.join(data_root, scenario_name)
        self.inputs = torch.from_numpy(np.load(f"{folder}/X_{mode}.npy")).float()
        self.labels = torch.from_numpy(np.load(f"{folder}/y_{mode}.npy")).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]