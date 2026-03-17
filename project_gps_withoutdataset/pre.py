"""
1. 修改预处理代码，一个是基站坐标，这个是固定的，另外一个是车的坐标，从 scenario31_dev 读取到车的坐标的路径的相对路径

这个相对路径和数据集真实存在的路径进行拼接就是每一帧数据车的GPS坐标的路径，然后就读取这个每一帧数据车的GPS信息

2. 有上述两个信息后，就可以进行特征工程

3. 特征工程：相对距离、相对方位，问一下AI，看一下需要提取哪些有用特征

4. 拆分数据集：训练集、验证集、测试集，按照6:2:2的比例进行划分

5. 现在需要做四个场景的训练，让 Ai 给出一个好的项目目录结构
"""


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# 从我们重构的 src 文件夹导入工具函数
from src.utils import gps_to_meters

# ========================================================
# 1. 路径配置 (全部使用相对路径)
# ========================================================
# 获取当前脚本 pre.py 所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据根目录: ./Data/raw
RAW_DATA_ROOT = os.path.join(BASE_DIR, "Data", "raw")
# 预处理后输出目录: ./Data/processed
PROCESSED_DATA_ROOT = os.path.join(BASE_DIR, "Data", "processed")

# 需要处理的场景列表 (以后可以增加 "scenario32" 等)
SCENARIOS = ["scenario32"]

# ========================================================
# 2. 辅助函数
# ========================================================
def read_gps_txt(file_path):
    """安全读取每一帧的GPS txt文件"""
    if not os.path.exists(file_path):
        return None, None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                lat = float(lines[0].strip())
                lon = float(lines[1].strip())
                if np.isnan(lat) or np.isnan(lon):
                    return None, None
                return lat, lon
    except:
        return None, None
    return None, None

# ========================================================
# 3. 主预处理逻辑
# ========================================================
def process_scenario(scenario_name):
    print(f"\n>>> 开始预处理场景: {scenario_name}")
    
    # 场景原始数据所在的文件夹: ./Data/raw/scenario31
    scenario_raw_path = os.path.join(RAW_DATA_ROOT, scenario_name)
    
    # --- 修改这里：兼容 scenario31_dev.csv ---
    csv_filename = f"{scenario_name}_dev.csv" # 自动拼接成 scenario31_dev.csv
    csv_path = os.path.join(scenario_raw_path, csv_filename)
    
    # 如果找不到 _dev.csv，尝试找普通的 .csv
    if not os.path.exists(csv_path):
        csv_path = os.path.join(scenario_raw_path, f"{scenario_name}.csv")

    unit1_gps_path = os.path.join(scenario_raw_path, 'unit1', 'GPS_data', 'gps_location.txt')

    if not os.path.exists(csv_path):
        print(f"跳过: 在 {scenario_raw_path} 下找不到 CSV 文件")
        return
    
    print(f"成功定位 CSV: {csv_path}")

    # A. 读取基站(Unit 1)固定坐标
    u1_lat, u1_lon = read_gps_txt(unit1_gps_path)
    if u1_lat is None:
        print(f"跳过: 无法读取基站GPS {unit1_gps_path}")
        return
    print(f"基站坐标: Lat={u1_lat}, Lon={u1_lon}")

    # B. 读取CSV总表
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # 清洗列名空格

    all_features = []
    all_labels = []

    # C. 遍历每一行提取特征
    print("正在提取特征并执行特征工程...")
    for _, row in df.iterrows():
        # 获取车辆GPS文件的相对路径并转换
        # 例如将 './unit2/GPS_data/GPS_location_173.txt' 
        # 转换为 'unit2/GPS_data/GPS_location_173.txt'
        rel_path = row['unit2_loc'].replace('./', '')
        u2_gps_full_path = os.path.join(scenario_raw_path, rel_path)
        
        u2_lat, u2_lon = read_gps_txt(u2_gps_full_path)
        
        if u2_lat is not None:
            # --- 特征工程 ---
            # 1. 相对米制位移 (dx, dy)
            dx, dy = gps_to_meters(u2_lat, u2_lon, u1_lat, u1_lon)
            # 2. 欧氏距离 (Distance)
            dist = np.sqrt(dx**2 + dy**2)
            # 3. 相对方位角 (Angle in radians)
            angle = np.arctan2(dy, dx)
            # 4. 车辆速度 (Speed)
            speed = float(row['unit2_spd_over_grnd_kmph'])
            
            # 组合成 5 维特征向量
            feature_vector = [dx, dy, dist, angle, speed]
            
            # 检查是否有非法数值
            if not np.isnan(feature_vector).any():
                all_features.append(feature_vector)
                # 标签: 波束编号 1-64 转为 0-63
                all_labels.append(int(row['unit1_beam']) - 1)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # D. 数据集划分 (6:2:2)
    # 第一次划分：分出 60% 训练集，剩下 40% 为临时集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=None
    )
    # 第二次划分：将 40% 的临时集平分为 验证集(20%) 和 测试集(20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
    )

    # E. 归一化 (基于训练集计算均值和标准差)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # 防止除以0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # F. 保存结果
    output_scenario_dir = os.path.join(PROCESSED_DATA_ROOT, scenario_name)
    os.makedirs(output_scenario_dir, exist_ok=True)

    # 保存 6 个核心文件
    np.save(os.path.join(output_scenario_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_scenario_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_scenario_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_scenario_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_scenario_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_scenario_dir, 'y_test.npy'), y_test)
    
    # 保存归一化参数（以后测试单条数据时有用）
    np.save(os.path.join(output_scenario_dir, 'scaler_params.npy'), {'mean': mean, 'std': std})

    print("-" * 30)
    print(f"场景 {scenario_name} 处理成功！")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"数据已保存至: {output_scenario_dir}")

# ========================================================
# 4. 执行入口
# ========================================================
if __name__ == "__main__":
    for scenario in SCENARIOS:
        process_scenario(scenario)