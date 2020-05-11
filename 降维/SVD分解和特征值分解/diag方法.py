# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : diag方法.py
# Time       ：2020/3/16 21:13 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np
# 输入是1维数组时，结果形成一个以一维数组为对角线元素的矩阵
data_1 = np.array([1,2,3])
print("输入是1维数组时的结果：\n",np.diag(data_1))

# 输入是二维矩阵时，结果输出矩阵的对角线元素
data_2 = np.array([[1,0,4],[2,2,0],[0,0,5]])
print("输入是2维数组时的结果：\n",np.diag(data_2))