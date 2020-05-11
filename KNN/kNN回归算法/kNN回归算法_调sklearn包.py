# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : kNN回归算法_调sklearn包.py
# Time       ：2020/3/29 20:54 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

def get_best_K(train_X, train_y,K):
    """获取最佳K值"""
    # 构建空的列表，用于存储平均MSE
    mse = []
    for k in K:
        # 使用10重交叉验证的方法，比对每一个k值下KNN模型的计算MSE
        # weights：用于指定近邻样本的投票权重，默认为’uniform’，表示所有近邻样本的投票权重一样；如果为’distance’，则表示投票权重与距离成反比，即近邻样本与未知类别的样本点距离越远，权重越小，反之，权值越大
        kNN_regressor = KNeighborsRegressor(n_neighbors= int(k), weights = 'distance')
        cv_result = cross_val_score(kNN_regressor, train_X, train_y, cv = 10, scoring= 'neg_mean_squared_error')  # 出来的值都是负值
        mse.append((- 1*cv_result).mean())

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

def kNN_regressor_by_sklearn(train_X, train_y,test_X, K):
    """调用KNN回归算法包实现"""
    # 从训练数据集中获取最佳k值
    best_K = get_best_K(train_X, train_y, K)
    # 重新构建模型，并将最佳的近邻个数设置为
    knn_reg = KNeighborsRegressor(n_neighbors=best_K, weights='distance')
    # 模型拟合
    knn_reg.fit(train_X, train_y)
    # 模型在测试集上的预测
    predict = knn_reg.predict(test_X)
    return predict


if __name__ == '__main__':
    # 读入数据
    X, y = load_data()
    # 设置待测试的不同k值
    K = np.arange(1, np.ceil(np.log2(X.shape[0])))
    # 训练集与测试集数据量比7:3 训练集用来训练结果 测试集测试
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state=3)
    predict_by_sklearn = kNN_regressor_by_sklearn(train_X, train_y, test_X, K)
    # 计算MSE值
    mse_test = mean_squared_error(test_y, predict_by_sklearn)
    print("测试集最终的mse:", mse_test)
    # 对比真实值和实际值
    test_pre = pd.DataFrame({'Real': test_y, 'Predict': predict_by_sklearn})
    print(test_pre.head(10))
