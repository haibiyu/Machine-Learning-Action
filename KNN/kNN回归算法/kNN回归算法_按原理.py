# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : kNN回归算法_按原理.py
# Time       ：2020/3/29 22:49 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
def load_data():
    """
    获取数据，并对数据进行归一化

    前4个变量为自变量，AT表示高炉的温度、V表示炉内的压力、AP表示高炉的相对湿度、RH表示高炉的排气量；
    最后一列为连续型的因变量，表示高炉的发电量
    :return:
    """
    ccpp = pd.read_excel('CCPP.xlsx')
    print(ccpp.head())
    print(ccpp.shape)
    # 由于4个自变量的量纲不一致,所以要进行归一化
    from sklearn.preprocessing import minmax_scale
    y = ccpp['PE']
    X = pd.DataFrame(minmax_scale(ccpp.iloc[:,:-1]),columns=ccpp.columns[:-1])
    print(X.head())
    return X, y

def kNN_regressor(X, dataSet, labels, k):
    """
    k近邻回归算法
    :param X: 需要预测类别的点
    :param dataSet: 特征数据集
    :param labels: 标签集
    :param k: 输入的k为正整数
    :return:
    """
    # 计算样本X与dataSet之间的距离
    m = len(dataSet)
    diffMat = np.tile(X, (m, 1)) - dataSet
    sqare_diffMat = diffMat ** 2
    sum_sqare_diffMat = sqare_diffMat.sum(axis=1)
    distances = np.sqrt(sum_sqare_diffMat)
    soretedDistIndex = np.array(distances.argsort())
    yVals = []
    for i in range(k):
        yVal = labels.iloc[soretedDistIndex[i]]
        yVals.append(yVal)
    predict_y = np.array(yVals).mean()
    return predict_y

def get_best_K(train_X, train_y,K):
    """获取最佳K值"""
    # 将训练集数据拆分为训练集和验证集,训练集与测试集数据量比7:3 训练集用来训练结果 测试集测试
    train_x, vari_x, train_y, vari_y = train_test_split(train_X,train_y, test_size=0.3,random_state=3)
    # 构建空的列表，用于存储平均MSE
    mse = []
    K = np.arange(1,int(K))
    for k in K:
        predict_vals=[]
        for i in range(len(vari_x)):
            predict_value = kNN_regressor(vari_x.iloc[0,:], train_x, train_y, k)
            predict_vals.append(predict_value)
        mse_k =(np.sum((predict_vals-vari_y)**2))/len(vari_x)
        mse.append(mse_k)
    # 从k个平均MSE中挑选出最小值所对应的下标
    arg_min = np.array(mse).argmin()
    # 绘制不同K值与平均MSE之间的折线图
    plt.plot(K, mse)
    # 添加点图
    plt.scatter(K, mse)
    # 添加文字说明
    plt.text(K[arg_min], mse[arg_min] + 0.5, '最佳k值为%s' % int(K[arg_min]))
    # 显示图形
    plt.show()
    return int(K[arg_min])

def kNN_regressor_by_theory(train_X, train_y,test_X, K):
    """按KNN回归算法原理实现"""
    # 从训练数据集中获取最佳k值
    best_K = get_best_K(train_X, train_y,K)
    # 预测数据
    result = []
    for i in range(len(test_X)):
        predict = kNN_regressor(test_X.iloc[i,:], train_X, train_y, best_K)
        result.append(predict)
    return result


if __name__ == '__main__':
    # 读入数据
    X, y = load_data()
    # 设置待测试的不同k值
    K = np.ceil(np.log2(X.shape[0]))
    # 训练集与测试集数据量比7:3 训练集用来训练结果 测试集测试
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state=3)
    predict_by_theory = kNN_regressor_by_theory(train_X, train_y, test_X, K)
    # 计算MSE值
    mse_test = (np.sum((predict_by_theory - test_y) ** 2)) / len(test_y)
    print("测试集最终的mse:", mse_test)
    # 对比真实值和实际值
    test_pre = pd.DataFrame({'Real': test_y, 'Predict': predict_by_theory})
    print(test_pre.head(10))

