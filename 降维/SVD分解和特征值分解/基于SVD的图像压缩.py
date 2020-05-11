# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 基于SVD的图像压缩.py
# Time       ：2020/3/18 22:01 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
# def printMat(inMat,thresh=0.8):
#     for i in range(32):
#         for k in range(32):
#             if float(inMat[i,k] > thresh):
#                 print(1)
#             else:
#                 print(0)
#         print('')
#
# def imgCompress(numSV=3,thresh=0.8):
#     myl=[]
#     pass

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def get_k(Sigma):
    print(Sigma)
    print("90%的总能量为：",sum(Sigma) * 0.9)
    for i in range(len(Sigma)):
        if sum(Sigma[:i]) > sum(Sigma) * 0.9:
            print("总能量高于90%的最小k为:",i)
            print("前k个总能量为",sum(Sigma[:i]))
            return i

if __name__ == '__main__':

    path = 'Andrew Ng.jpg'
    data = io.imread(path,as_grey=True)
    print(data.shape)
    data = np.mat(data)  # 需要mat处理后才能在降维中使用矩阵的相乘
    U, sigma, VT = np.linalg.svd(data)
    # # 在重构之前，依据前面的方法需要选择达到某个能量度的奇异值
    # # cnt = sum(sigma)
    # # print(cnt)
    # # cnt90 = 0.9 * cnt  # 达到90%时的奇异总值
    # # print(cnt90)
    count = 30  # 选择前30个奇异值
    # # cntN = sum(sigma[:count])
    # # print(cntN)
    # count = get_k(sigma)

    # 重构矩阵
    dig = np.diag(sigma[:count])  # 获得对角矩阵
    # dim = data.T * U[:,:count] * dig.I # 降维 格外变量这里没有用  dig.I：是求dig的逆矩阵
    redata = U[:, :count] * dig * VT[:count, :]  # 重构后的数据

    plt.imshow(data, cmap='gray')  # 取灰
    plt.title("原始图片")
    plt.show()
    plt.imshow(redata, cmap='gray')  # 取灰
    plt.title("基于SVD压缩重构后的图片")
    plt.show()

