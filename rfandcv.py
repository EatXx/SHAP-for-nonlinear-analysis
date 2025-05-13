# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:40:57 2025

@author: DinoChen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:48:58 2024

@author: JOJOChen
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 读取数据
data = pd.read_excel('grid.xlsx') # 文件名
X = data.drop(columns=['patentk','Join_Count']) # 非特征列
y = data['patentk']

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. 创建随机森林回归模型
rf_model = RandomForestRegressor()

# 4. 定义参数网格进行网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 5. 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(rf_model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 6. 输出最佳参数组合
best_params = grid_search.best_params_
print("最佳参数组合：", best_params)

# 7. 使用最佳参数训练随机森林回归模型
best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(X_train, y_train)

# 8. 使用交叉验证评估模型性能
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"交叉验证得分（MSE）：{cv_scores}")

# 9. 输出交叉验证结果的平均值
print(f"交叉验证 MSE 平均值: {-np.mean(cv_scores)}")

# 10. 预测并评估模型性能
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

n = len(y_test)
p = X_test.shape[1]
test_adjusted_r2 = 1 - (1 - r2_score(y_test, y_test_pred)) * ((n - 1) / (n - p - 1))
train_adjusted_r2 = 1 - (1 - r2_score(y_train, y_train_pred)) * ((n - 1) / (n - p - 1))

print(f"训练集 MSE: {train_mse}")
print(f"测试集 MSE: {test_mse}")
print(f"测试集调整后的 R^2: {test_adjusted_r2}")
print(f"训练集调整后的 R^2: {train_adjusted_r2}")