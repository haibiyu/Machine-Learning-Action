# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 调用SVD和eig方法.py
# Time       ：2020/3/16 21:26 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np

data = np.array([[1, 0, 4], [2, 2, 0], [0, 0, 5]])  # 数组
# 调用np.linalg.eig（）对data*data'进行特征值分解
eig_value, eig_vector = np.linalg.eig(data.dot(data.T))
# 将特征值降序排列
eig_value = np.sort(eig_value)[::-1]
print("特征值：", eig_value)  # 特征值
print("特征值的平方根：", np.sqrt(eig_value))  # 特征值的平方根

# 调用np.linalg.svd（）对data进行奇异值分解
U, S, V = np.linalg.svd(data)  # S:[6.43771974 2.87467944 0.54035417] 前两个数值较大
## np.diag(array)参数说明：
## array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵；
## array是一个二维矩阵时，结果输出矩阵的对角线元素

# 将数据降到二维后，计算U[:,:2]*S[:2]*V[:2,:] 结果应该近似于data
recon_data = np.round(U[:,:2].dot(np.diag(S[:2])).dot(V[:2,:]), 0).astype(int)
print("奇异值：", S)  # 奇异值
print("data:\n",data)
print("U*S*V的结果:\n",recon_data)
print(U,V)