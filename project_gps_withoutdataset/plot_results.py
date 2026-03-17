import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(scenario='scenario31'):
    log_path = f'./logs/{scenario}_train_log.csv'
    if not os.path.exists(log_path):
        print("未找到日志文件，请先运行 train.py")
        return

    df = pd.read_csv(log_path)

    plt.figure(figsize=(12, 5))

    # 图1：损失曲线 (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title(f'{scenario} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 图2：准确率曲线 (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_top1'], label='Val Top-1 Acc', color='green')
    plt.plot(df['epoch'], df['val_top5'], label='Val Top-5 Acc', color='orange')
    plt.title(f'{scenario} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'./logs/{scenario}_curves.png') # 保存图片
    plt.show()

if __name__ == "__main__":
    plot_training_results('scenario32')