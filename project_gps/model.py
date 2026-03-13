import torch.nn as nn

class BeamMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(BeamMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), # 输入层 (3 -> 128)
            nn.ReLU(),
            nn.Linear(128, 256),        # 隐藏层 (128 -> 256)
            nn.ReLU(),
            nn.Dropout(0.2),           # 随机关闭20%神经元，防止过拟合
            nn.Linear(256, 128),        # 隐藏层 (256 -> 128)
            nn.ReLU(),
            nn.Linear(128, output_dim)  # 输出层 (128 -> 64个波束)
        )
        
    def forward(self, x):
        return self.network(x)