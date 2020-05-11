# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 利用sklearn中PCA包进行降维.py
# Time       ：2020/3/15 9:48 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：
"""
from sklearn.decomposition import PCA #PCA模块
from sklearn.decomposition import KernelPCA #核PCA模块
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def make_blobs_data():
    # X为样本数据，Y为簇类别， 共10000个样本，每个样本3个特征，共4个簇
    X, y = make_blobs(n_samples=10000, n_features=3,
                      centers=[[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
                      cluster_std=[0.1, 0.2, 0.1, 0.2],
                      random_state=3)
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
    plt.title("原始数据")
    plt.show()
    return X

def PCA_main():
    # 获取数据
    data_mat = make_blobs_data()  # 随机生成的数据
    # PCA只适用于密集型数据，不适合大尺寸的数据
    # n_components值是大于等于1的整数，则表示降维后的特征数
    # n_components值在(0，1]之间,则表示主成分的方差和所占的最小比例阈值
    # n_components='mle':自动确定保留的特征数
    pca = PCA(n_components=2).fit(data_mat)
    # explained_variance_ratio_返回降维后的各主成分的方差值占总方差值的比例,即单个变量方差贡献率，这个比例越大，则越是重要的主成分
    print(pca.explained_variance_ratio_)
    # 降维后的数据
    reduced_X = pca.transform(data_mat)

    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], marker='o')
    plt.title("PCA")
    plt.show()

def make_circles_data():
    """生成圆形的二维数据"""
    # factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点
    x, y = make_circles(n_samples=600, factor=0.2, noise=0.02)  # factor代表维度
    return x,y

def KPCA_main():
    x, y = make_circles_data()
    # 使用kernal PCA
    # 这里调用的核是径向基函数（Radial Basis Function, RBF）
    # gamma是一个核参数（用于处理非线性）
    kpca = KernelPCA(kernel='rbf', gamma=10)
    x_kpca = kpca.fit_transform(x)
    # 绘图
    plot_kpca_picture(x, y, x_kpca)

def plot_kpca_picture(x,y,x_kpca):
    plt.subplot(121)
    plt.title('原始数据空间')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.subplot(122)
    plt.title('KPCA')
    plt.scatter(x_kpca[:, 0], x_kpca[:, 1], c=y)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

if __name__ == '__main__':
    # PCA_main()
    KPCA_main()


