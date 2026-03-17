"""
训练过程需要保存的内容：
训练日志：每个epoch的训练损失、验证损失、验证准确率，方便后续进行可视化，模型权重也保存

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd # 用于保存日志

# 导入我们重构的模块
from src.dataset import BeamDataset
from src.model import BeamMLP
from src.utils import calculate_topk_accuracy

def run_train(scenario='scenario31'):
    # --- 1. 参数与目录准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    # 日志文件路径
    log_path = f'./logs/{scenario}_train_log.csv'
    # 用于存放每一轮数据的列表
    history = []

    # --- 2. 数据加载 ---
    train_ds = BeamDataset(scenario, mode='train')
    val_ds = BeamDataset(scenario, mode='val')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # --- 3. 模型初始化 ---
    model = BeamMLP(input_dim=5, output_dim=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_val_acc = 0.0 # 记录最高准确率

    # --- 4. 训练循环 ---
    epochs = 100 # 增加轮数，反正有BN层学得快
    print(f"开始训练场景: {scenario} (设备: {device})")

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train() # 启用 BatchNorm 和 Dropout
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval() # 禁用 BatchNorm 的更新和 Dropout
        val_loss = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # 计算 Top-1 和 Top-5
                t1, t5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
                val_top1 += t1.item()
                val_top5 += t5.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_top1 / len(val_loader)
        avg_val_top5 = val_top5 / len(val_loader)

        # --- 5. 记录日志 ---
        print(f"Epoch {epoch+1:03d} | TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ValTop1: {avg_val_acc:.2f}% | ValTop5: {avg_val_top5:.2f}%")
        
        # 将结果存入 list
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_top1': avg_val_acc,
            'val_top5': avg_val_top5
        })

        # --- 6. 保存最佳权重 ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), f"./checkpoints/{scenario}_best.pth")
            print(f"--> 检测到更好的模型，已保存至 checkpoints (Acc: {best_val_acc:.2f}%)")

    # --- 7. 保存最终日志到 CSV ---
    df_history = pd.DataFrame(history)
    df_history.to_csv(log_path, index=False)
    print(f"\n训练结束！日志已保存至 {log_path}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

if __name__ == "__main__":
    run_train('scenario32')