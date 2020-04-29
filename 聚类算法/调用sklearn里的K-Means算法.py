# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 调用sklearn里的K-Means算法.py
# Time       ：2020/3/6 15:48 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：
"""

import time
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_iris_data():
    """
    获取鸢尾花数据集
    特征分别是sepal length、sepal width、petal length、petal width
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, 2:]  # 通过花瓣的两个特征来聚类
    k = 3  # 假设聚类为3类
    return X,k

def load_data_make_blobs():
    """
    生成模拟数据
    """""
    from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
    # 生成模拟数据
    k = 5  # 给定聚类数量
    X, Y = make_blobs(n_samples=1000, n_features=2, centers=k, random_state=1)
    return X,k

if __name__ == '__main__':

    # X, k = load_iris_data()  # 获取鸢尾花数据和聚类数量
    X, k = load_data_make_blobs()  # 获取模拟数据和聚类数量
    # 构建模型
    s = time.time()
    km = KMeans(n_clusters=k)
    km.fit(X)
    print("用sklearn内置的K-Means算法聚类耗时：", time.time() - s)

    label_pred = km.labels_  # 获取聚类后的样本所属簇对应值
    centroids = km.cluster_centers_  # 获取簇心

    # 绘制K-Means结果
    # 未聚类前的数据分布
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    # plt.xlabel('petal length')
    # plt.ylabel('petal width')
    plt.title("未聚类前的数据分布")
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=label_pred, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=100)
    # plt.xlabel('petal length')
    # plt.ylabel('petal width')
    plt.title("用sklearn内置的K-Means算法聚类结果")
    plt.show()