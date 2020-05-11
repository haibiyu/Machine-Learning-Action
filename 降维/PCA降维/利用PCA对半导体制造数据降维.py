# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 利用PCA对半导体制造数据降维.py
# Time       ：2020/3/14 21:45 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data_set(file_name,delim='\t'):
    """
    获取数据
    :param file_name: 文件路径
    :param delim: 分隔符
    :return: 返回处理后的数据
    """
    fr = open(file_name)
    string_arr = [line.strip().split(delim) for line in fr.readlines()]
    data_arr = [np.array(line).astype(float) for line in string_arr]
    return np.mat(data_arr)

def replace_nan_with_mean():
    """
    将数据中NaN值替换为平均值
    """
    data_mat = load_data_set('./半导体制造数据/secom.data',' ')
    numFeat = data_mat.shape[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        mean_val = np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:, i].A))[0], i])
        # 将所有NaN值设置为平均值
        data_mat[np.nonzero(np.isnan(data_mat[:,i].A))[0], i] = mean_val
    return data_mat

def pca(data_mat, variance_ratio=0.99):
    """
    利用PCA对数据进行降维，获取降维后的数据和重构后的数据
    :param data_mat: 原始数据，m*n的矩阵
    :param top_k_feat: 需要降到的维度数
    :return:
    """
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = (data_mat - mean_vals)  # 去均值化
    cov_mat = np.cov(mean_removed, rowvar=0)  # 计算协方差矩阵 n*n
    # 通常用奇异值分解SVD 代替 特征值分解eig
    U, S, V = np.linalg.svd(cov_mat)  # 获得SVD后的 U(n*n)、S(n*n)、V(n*n)，特征值S已降序排列

    # 获取保留方差99%的最小维度数top_k_feat
    top_k_feat = get_top_k_feat(S, variance_ratio)
    print("降维后保留方差{}的最小维度数为：{}".format(variance_ratio,top_k_feat))
    plot_top_variance_ratio(S, top_k_feat)

    red_vects = U[:, :top_k_feat]  # 取前top_k_feat列的特征向量
    red_data_mat = mean_removed * red_vects  # 将原始数据转换到降维后的空间上
    recon_mat = red_data_mat * red_vects.T + mean_vals  # 重构原始数据
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

def plot_top_variance_ratio(eigvalues,k):
    """
    绘制前k个主成分占总方差的百分比
    :param eigvalues:特征值
    :param k:降维后的维度数目
    """
    plt.plot(np.arange(1, k+1), eigvalues[:k] / np.sum(eigvalues) * 100,'o-')
    plt.xlabel("主成分数目")
    plt.ylabel("方差的百分比")
    plt.xlim(0, k)
    plt.ylim(0,)
    plt.title("前{}个主成分占总方差的百分比".format(k))
    plt.show()

if __name__ == '__main__':
    # 对数据进行处理
    data_mat = replace_nan_with_mean()
    # 获取降维后的数据和重构后的数据
    red_data_mat, recon_mat = pca(data_mat,0.99)
