# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 基于原理实现SVD分解.py
# Time       ：2020/3/16 21:13 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np

data = np.array([[1, 0, 4], [2, 2, 0], [0, 0, 5]])
# 1.调用np.linalg.eig（）计算特征值和特征向量
eig_val, u_vec = np.linalg.eig(data.dot(data.T))
s_val = np.sqrt(eig_val)  # 奇异值：是特征值的平方根
# 将向量u_vec对应排好序
s_val_sort_idx = np.argsort(s_val)[::-1]
u_vec = u_vec[:, s_val_sort_idx]
# 奇异值降序排列
s_val = np.sort(s_val)[::-1]

# 2.计算奇异值矩阵的逆
s_val_inv = np.linalg.inv(np.diag(s_val))
# 3.计算右奇异矩阵
v_vec = s_val_inv.dot((u_vec.T).dot(data))

# 降到两维，计算U*S*V 结果应该和data几乎相同
recon_data = np.round(u_vec[:,:2].dot(np.diag(s_val[:2])).dot(v_vec[:2,:]), 0).astype(int)
print("左奇异矩阵U:\n", u_vec)
print("奇异值Sigma:\n", s_val)
print("右奇异矩阵V:\n", v_vec)
print("data:\n",data)
print("U*S*V的结果:\n",recon_data)
