# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 主成分分析PCA.py
# Time       ：2020/3/14 20:23 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def random_data():
    """随机生成1000个点，围绕在y=0.1x+0.1的直线周围"""
    num_points = 1000
    data_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.5)  # 正态分布 均值为0.0  标准差为0.5
        y1 = x1 * 0.1 + 0.1 + np.random.normal(0.0, 0.03)
        data_set.append([x1, y1])
    data_mat = np.mat(data_set)
    return data_mat


def pca(data_mat, top_k_feat=9999999):
    """
    利用PCA对数据进行降维，获取降维后的数据和重构后的数据
    :param data_mat: 原始数据，m*n的矩阵
    :param top_k_feat: 需要降到的维度数
    :return:
    """
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = (data_mat - mean_vals)  # 去均值化
    # 数据缩放:对特征之间数量级存在较大差异的，去量钢化
    # stddev_vals = np.std(data_mat, axis=0)
    # mean_removed = mean_removed / stddev_vals
    # mean_removed = np.mat(pd.DataFrame(mean_removed).replace(np.nan, 0))
    cov_mat = np.cov(mean_removed, rowvar=0)  # 计算协方差矩阵 n*n
    # 通常用奇异值分解SVD 代替 特征值分解eig
    U, S, V = np.linalg.svd(cov_mat)  # 获得SVD后的 U(n*n)、S(n*n)、V(n*n)，特征值S已降序排列
    red_vects = U[:, :top_k_feat]  # 取前top_k_feat列的特征向量
    red_data_mat = mean_removed * red_vects  # 将原始数据转换到降维后的空间上
    recon_mat = red_data_mat * red_vects.T + mean_vals  # 重构原始数据
    # recon_mat = np.mat(
    #     (red_data_mat * red_vects.T).A * (stddev_vals.A)) + mean_vals  # 重构原始数据
    return red_data_mat, recon_mat


def get_top_k_feat(eig_values,variance_ratio=0.99):
    """
    根据variance_ratio确定保留的特征数
    :param eig_values: 特征值，从大到小排序
    :param variance_ratio: 主成分的方差和所占的最小比例阈值
    :return:
    """
    sum_S = float(np.sum(eig_values))
    curr_S = 0
    for i in range(len(eig_values)):
        curr_S += float(eig_values[i])
        if curr_S / sum_S >= variance_ratio:
            return i + 1


def plot_picture(dataMat, reconMat):
    plt.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0],
                c='b', marker='o', s=50)
    plt.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0],
                c='r', marker='^', s=30)
    plt.title("原始数据和重构之后的数据分布")
    plt.show()


if __name__ == '__main__':
    # 获取数据
    data_mat = random_data()  # 随机生成的数据

    # 获取降维后的数据和重构后的数据
    red_data_mat, recon_mat = pca(data_mat, 1)

    # 绘图
    plot_picture(data_mat, recon_mat)
