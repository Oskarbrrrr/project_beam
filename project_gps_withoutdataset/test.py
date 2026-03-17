import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# 导入你的模块
from src.dataset import BeamDataset
from src.model import BeamMLP
from src.utils import calculate_topk_accuracy

def run_test(scenario='scenario31'):
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在测试场景: {scenario} | 使用设备: {device}")

    # 2. 加载测试数据集 (mode='test')
    try:
        test_ds = BeamDataset(scenario, mode='test')
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        print(f"测试集加载成功，样本数: {len(test_ds)}")
    except FileNotFoundError:
        print("错误：找不到测试集文件，请确保已经运行了 pre.py 且 Data/processed 下有测试数据。")
        return

    # 3. 初始化模型并加载最佳权重
    model = BeamMLP(input_dim=5, output_dim=64).to(device)
    checkpoint_path = f"./checkpoints/{scenario}_best.pth"
    
    if os.path.exists(checkpoint_path):
        # 加载保存的状态字典
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"已成功加载最佳模型权重: {checkpoint_path}")
    else:
        print(f"错误：找不到模型权重文件 {checkpoint_path}")
        return

    # 4. 进入评估模式
    model.eval()
    
    test_top1 = 0.0
    test_top5 = 0.0
    total_samples = 0

    print("正在评估测试集...")
    print("-" * 30)

    with torch.no_grad(): # 测试时不需要计算梯度
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算 Top-1 和 Top-5 准确率
            t1, t5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            
            # 这里的 t1 和 t5 是当前 batch 的平均值，需要乘以 batch 大小还原成个数
            batch_size = targets.size(0)
            test_top1 += t1.item() * batch_size
            test_top5 += t5.item() * batch_size
            total_samples += batch_size

    # 5. 计算最终平均准确率
    final_top1 = test_top1 / total_samples
    final_top5 = test_top5 / total_samples

    print("-" * 30)
    print(f"测试完成！")
    print(f"最终测试集准确率 (Test Set Accuracy):")
    print(f">> Top-1 Acc: {final_top1:.2f}%")
    print(f">> Top-5 Acc: {final_top5:.2f}%")
    print("-" * 30)

    # 记录到文本文件
    result_path = f"./logs/{scenario}_final_test_result.txt"
    with open(result_path, 'w') as f:
        f.write(f"Scenario: {scenario}\n")
        f.write(f"Final Top-1 Acc: {final_top1:.2f}%\n")
        f.write(f"Final Top-5 Acc: {final_top5:.2f}%\n")
    print(f"最终评估结果已记录至: {result_path}")

if __name__ == "__main__":
    run_test('scenario32')