# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:31:32 2024

@author: JOJOChen
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
data = pd.read_excel('grid.xlsx')  # 文件名
X = data.drop(columns=['Join_Count','patentk','Vmean'])  # 非特征列
y = data['patentk']

# 2. 创建并训练rft模型
rf_params =  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X, y)

# 3. 创建SHAP解释器
explainer = shap.Explainer(rf_model)

# 4. 计算SHAP值

shap_interaction_values = explainer.shap_interaction_values(X)


# 计算平均交互值矩阵
average_interaction_values = np.mean(shap_interaction_values, axis=0)


# 为每个特征找到交互最强的另一个特征
strongest_interaction_indices = []
for i in range(len(X.columns)):
    # 忽略对角线元素
    interaction_strengths = np.abs(average_interaction_values[i, :].copy())
    # 使用 argsort 获取从小到大的索引排序
    sorted_indices = np.argsort(interaction_strengths)

    # 第二大值的索引是倒数第二个
    second_largest_index = sorted_indices[-2]


    # 找到最强交互的索引
    strongest_interaction_index = second_largest_index
    strongest_interaction_indices.append(strongest_interaction_index)

# 绘制每个特征及其最强交互特征的SHAP依赖图
for i, interaction_index in enumerate(strongest_interaction_indices):
    feature_name = X.columns[i]  # 当前特征
    interaction_feature_name = X.columns[interaction_index]  # 与之交互最强的特征
    
    shap.dependence_plot((feature_name, interaction_feature_name), shap_interaction_values, X, display_features=X)