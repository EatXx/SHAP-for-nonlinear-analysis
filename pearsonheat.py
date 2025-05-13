# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:31:07 2023

@author: JOJOChen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件

df = pd.read_excel('street_xclean重命名.xlsx') # 替换成你的Excel文件路径


# 计算Pearson相关系数
correlation_matrix = df.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Pearson Correlation Heatmap')
plt.show()