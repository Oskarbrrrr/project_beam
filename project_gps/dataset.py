import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BeamDataset(Dataset):
    def __init__(self, data_dir='./Data'):
        x_path = os.path.join(data_dir, 'X_train.npy')
        y_path = os.path.join(data_dir, 'y_train.npy')
        
        self.inputs = torch.from_numpy(np.load(x_path)).float()
        self.labels = torch.from_numpy(np.load(y_path)).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]