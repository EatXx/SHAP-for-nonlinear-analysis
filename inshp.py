# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:23:31 2024

@author: JOJOChen
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel('spot3grid.xlsx')  # 文件名
X = data.drop(columns=['Join_Count','patentk','Vmean'])  # 非特征列
y = data['patentk']

# 2. 创建并训练rf模型
rf_params =  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X, y)

# 3. 创建SHAP解释器
explainer = shap.Explainer(rf_model)

# 4. 计算SHAP值
shap_values = explainer.shap_values(X)
# shap_interaction_values = explainer.shap_interaction_values(X)

# 5. 生成特征重要性的条形图
shap.summary_plot(shap_values, X, plot_type="bar")
plt.show()


# 6. 生成每个特征的SHAP值偏相关图
for feature_name in X.columns:
    shap.dependence_plot(feature_name, shap_values, X, interaction_index=None)

