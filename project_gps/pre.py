import pandas as pd
import numpy as np
import os

CSV_PATH = r'E:\dataset\scenario31_new\scenario31\scenario31_dev.csv'
UNIT1_GPS_FILE = r'E:\dataset\scenario31_new\scenario31\unit1\GPS_data\gps_location.txt'
UNIT2_GPS_DIR = r'E:\dataset\scenario31_new\scenario31\unit2\GPS_data'
OUTPUT_DIR = './Data'

def read_gps_txt(file_path):
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

def preprocess():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    u1_lat, u1_lon = read_gps_txt(UNIT1_GPS_FILE)
    if u1_lat is None:
        print("错误：无法读取基站GPS文件或文件内容有误！")
        return

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    all_features = []
    all_labels = []

    
    count_total = 0
    count_nan = 0

    for i, row in df.iterrows():
        count_total += 1
        idx = int(row['index'])
        u2_file_path = os.path.join(UNIT2_GPS_DIR, f"GPS_location_{idx}.txt")
        
        u2_lat, u2_lon = read_gps_txt(u2_file_path)
        
        try:
            speed = float(row['unit2_spd_over_grnd_kmph'])
            beam = int(row['unit1_beam'])
        except:
            speed, beam = np.nan, np.nan

        if (u2_lat is not None) and (not np.isnan(u2_lat)) and \
           (not np.isnan(u2_lon)) and (not np.isnan(speed)) and \
           (not np.isnan(beam)):
            
            d_lat = u2_lat - u1_lat
            d_lon = u2_lon - u1_lon
            
            all_features.append([d_lat, d_lon, speed])
            all_labels.append(beam - 1) # 转为 0-63
        else:
            count_nan += 1

    X = np.array(all_features)
    y = np.array(all_labels)

    if np.isnan(X).any():
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0 
    
    X_scaled = (X - X_mean) / X_std

    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_scaled.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y.astype(np.int64))

    print("-" * 30)
    print(f"总记录数: {count_total}")
    print(f"因包含NaN被剔除的记录数: {count_nan}")
    print(f"最终有效样本数: {len(X_scaled)}")
    print(f"最终特征矩阵是否有NaN: {np.isnan(X_scaled).any()}")
    print("-" * 30)

if __name__ == "__main__":
    preprocess()