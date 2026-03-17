"""
增加一下 批归一化层 ，看看能不能提升模型的性能
"""
import torch.nn as nn

class BeamMLP(nn.Module):
    def __init__(self, input_dim=5, output_dim=64):
        super(BeamMLP, self).__init__()
        
        self.network = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),       # 对 256 个神经元的输出做归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),       # 对 512 个神经元的输出做归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # 输出层
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)