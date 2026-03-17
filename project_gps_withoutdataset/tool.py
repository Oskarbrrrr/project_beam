import numpy as np
X = np.load('./Data/X_train.npy')
y = np.load('./Data/y_train.npy')
print(f"输入特征是否有NaN: {np.isnan(X).any()}")
print(f"标签是否有NaN: {np.isnan(y).any()}")
print(f"输入特征的最大值: {np.max(X)}")
print(f"输入特征的最小值: {np.min(X)}")