# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 00:26:34 2024

@author: JOJOChen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 读取数据
data = pd.read_excel('grid.xlsx') # 文件名
X = data.drop(columns=['patentk','Join_Count']) # 非特征列
y = data['patentk']

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. 创建多元线性回归模型
linear_model = LinearRegression()

# 4. 训练多元线性回归模型
linear_model.fit(X_train, y_train)

# 5. 在训练集和测试集上进行预测
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# 6. 计算均方误差 (MSE)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 7. 计算调整后的决定系数 (Adjusted R-squared)
n_train = len(y_train)
n_test = len(y_test)
p = X_test.shape[1]

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

adjusted_r2_train = 1 - (1 - train_r2) * ((n_train - 1) / (n_train - p - 1))
adjusted_r2_test = 1 - (1 - test_r2) * ((n_test - 1) / (n_test - p - 1))

print(f"训练集 MSE: {train_mse}")
print(f"测试集 MSE: {test_mse}")
print(f"训练集调整后的 R^2: {adjusted_r2_train}")
print(f"测试集调整后的 R^2: {adjusted_r2_test}")