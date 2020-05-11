# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 餐馆菜肴推荐_推荐未尝过的菜肴.py
# Time       ：2020/3/18 16:04 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：能够寻找用户没有尝过的菜肴，然后通过SVD减少特征空间并提高推荐的效果
"""
import numpy as np


def loadExData():
    return [[1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            ]


# 相似度的计算，都是采用列向量方式表示，暗示后续是基于物品的相似度计算

def ecludSim(A, B):
    """
    计算欧式距离相似度，相似度=1/(1+欧氏距离），相似度越大越好；距离为0时，相似度为1,
    """
    return 1.0 / (1.0 + np.linalg.norm(A - B))


def pearsSim(A, B):
    """
    计算pearson相似度，pearson相关系数取值范围在【-1,1】之间，将结果归一化到【0,1】之间
    """
    if len(A) < 3:  # 检查是否存在3个或更多点，如果不存在，返回1.0，这是因为两个向量完全相关
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(A, B, rowvar=0)[0][1]


def cosSim(A, B):
    """
    计算余弦相似度，余弦相似度范围在【-1,1】之间，将结果归一化到【0,1】之间
    """
    num = float(A.T * B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return 0.5 + 0.5 * (num / denom)  # 将数据归一化到【0,1】之间


def standEst(dataMat, user, simMeas, item):
    """
    给定相似度计算方法的条件下，用户对物品的估计评分值
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = dataMat.shape[1]  # 物品数
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 两个物品都同时都评级的数据索引
        overLap = \
            np.nonzero(np.logical_and(dataMat[:, item] > 0, dataMat[:, j] > 0))[
                0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is : %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if similarity == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    给user推荐前N个未评级物品
    :param dataMat: 用户-物品矩阵
    :param user:用户
    :param N:
    :param simMeas:
    :param estMethod:
    :return:
    """
    # 寻找未评过级的物品,[1]：对应的列索引
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[
           :N]  # 获取前N个未评级物品

# （用户x商品）    # 为0表示该用户未评价此商品，即可以作为推荐商品
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(3) * Sigma[:3])  # 建立对角矩阵
    xformedItem = dataMat.T * U[:, :3] * Sig4.I  # 构建转换后的物品
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 由于降维后的矩阵与原矩阵代表数据不同（行由用户变为了商品），所以在比较两件商品时应当取【该行所有列】 再转置为列向量传参
        similarity = simMeas(xformedItem[item, :].T, xformedItem[j, :].T)
        # print("the %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def print_sim_func():
    myMat = np.mat(loadExData())
    print("原始矩阵：\n", myMat)
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))

    print("余弦相似度：")
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))

    print("pearson相似度：")
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))

def get_k(dataMat):
    U, Sigma, VT = np.linalg.svd(dataMat)
    print(Sigma)
    sig2 = Sigma ** 2
    print("90%的总能量为：",sum(sig2) * 0.9)
    for i in range(len(sig2)):
        if sum(sig2[:i]) > sum(sig2) * 0.9:
            print("总能量高于90%的最小k为:",i)
            print("前k个总能量为",sum(sig2[:i]))
            return i



if __name__ == '__main__':
    # myMat = np.mat(loadExData())
    # myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    # myMat[3, 3] = 2
    # print("原始矩阵：\n", myMat)
    # print(recommend(myMat, 2))
    # print(recommend(myMat, 2, simMeas=ecludSim))
    # print(recommend(myMat, 2, simMeas=pearsSim))

    myMat = np.mat(loadExData2())
    print("原始矩阵：\n", myMat)
    print(recommend(myMat, 1,estMethod=svdEst))
    print(recommend(myMat, 1, estMethod=svdEst,simMeas=pearsSim))

    # k= get_k(myMat)
