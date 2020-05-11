# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : CART回归树算法.py
# Time       ：2020/4/21 8:20 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""

import numpy as np

def loadDataSet(fileName):
    """导入数据"""
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 使用python3会报错1，因为python3中map的返回类型是‘map’类，不能进行计算，需要将map转换为list
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    通过数组过滤切分数据集
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return:
    """
    # 使用python3会报错2，需要将书中脚本修改为以下内容
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """生成叶子节点，即目标变量的均值"""
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """计算数据集中目标变量的误差平方和
    误差平方和 = 目标变量的均方差 * 数据集的样本个数
    """
    return np.var(dataSet[:, -1]) * dataSet.shape[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    遍历所有的特征及其可能的取值来找到使误差平方和最小化的切分特征及其切分点
    :param dataSet: 数据集合
    :param leafType: 建立叶节点的函数,该参数也决定了要建立的是模型树还是回归树
    :param errType: 代表误差计算函数,即误差平方和计算函数
    :param ops: 用于控制函数的停止时机，第一个是容许的误差下降值，第二个是切分的最少样本数
    :return:最佳切分特征及其切分点
    """
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有值都相等，则停止切分，直接生成叶结点
    if len(set(dataSet[:-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    # 数据集中最后一列是标签，不是特征，所以这里是n-1
    for featIndex in range(n - 1):
        # set(dataSet[:,featIndex]) 使用python3会报错3，因为matrix类型不能被hash，需要修改为下面这句
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果切分出的数据集小于切分最小样本数，则继续下一个
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    # 如果误差减少不大，则停止切分，直接生成叶结点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    # 《机器学习实战》中，感觉下面这三句话多余(可以删了)，因为在上面已经判断过了切分出的数据集很小的情况 #
    # mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 用最佳切分特征和切分点进行切分
    # if mat0.shape[0] < tolN or mat1.shape[0] < tolN: # 如果切分出的数据集很小，则停止切分，直接生成叶结点
    #     return None, leafType(dataSet)

    return bestIndex, bestValue  # 返回最佳切分特征编号和切分点


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    构建CART回归树
    :param dataSet: 数据集，默认NumPy Mat
    :param leafType: 建立叶节点的函数,该参数也决定了要建立的是模型树还是回归树
    :param errType: 代表误差计算函数,即误差平方和计算函数
    :param ops: 用于控制函数的停止时机，第一个是容许的误差下降值，第二个是切分的最少样本数
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果feat为None, 则返回叶结点对应的预测值
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat  # 最佳切分特征
    retTree['spVal'] = val  # 切分点
    # 切分后的左右子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 后剪枝
def isTree(obj):
    """判断输入变量是否是一棵树，返回布尔类型的结果"""
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    """从上往下遍历树直到叶子节点为止，获取两个子节点的均值"""
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testData):
    """
    后剪枝
    :param tree: 待剪枝的树
    :param testData: 剪枝所需的测试数据
    :return:
    """
    # 如果没有测试数据，则对树进行塌陷处理
    if testData.shape[0] == 0:
        return getMean(tree)
    # 检查左右分支是否是子树还是叶节点
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果是子树，递归调用prune（）函数进行剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果左右分支都是叶节点，则计算并比较合并前后的误差
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(
            np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并后的误差小于不合并的误差，则进行合并操作，反之不进行合并操作
        if errorMerge < errorNoMerge:
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree


# 构建模型树
def linearSolve(dataSet):
    """
    将数据集格式化为目标变量Y和自变量X，并计算得到回归系数ws
    :param dataSet:
    :return: 回归系数ws,自变量X,目标变量Y
    """
    m, n = dataSet.shape
    # 将X和Y中的数据格式化
    X = np.mat(np.ones((m, n)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse,\n\
                         try increasing the second value od ops")
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    """负责生产叶节点的模型,返回回归系数ws"""
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """计算模型的平方误差和"""
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))

# 基于人的智力水平和自行车的速度的关系
def regTreeEval(model, inDat):
    """对回归树叶节点进行预测，返回一个预测值"""
    return float(model)


def modelTreeEval(model, inDat):
    """对于模型树叶节点进行预测，返回一个预测值"""
    n = inDat.shape[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    对于输入的单个数据点或者行向量进行预测，返回一个浮点值
    :param tree:生成树
    :param inData:输入的单个测试数据
    :param modelEval:指定数的类型，回归树还是模型树
    :return:
    """
    # 判断是否是树，如果不是，返回叶节点
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        # 判断是否有左子树，如果有递归进入子树，如果没有返回叶节点
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        # 判断是否有右子树，如果有递归进入子树，如果没有返回叶节点
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    """对整个测试集进行预测，返回一组预测值"""
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

def plot_data(dataMat):
    import matplotlib.pyplot as plt
    plt.scatter(np.array(dataMat[:, 0].T.tolist()),np.array(dataMat[:, 1].T.tolist()))
    plt.show()

if __name__ == '__main__':
    # dataMat = loadDataSet('./data/ex0.txt')
    # dataMat = np.mat(dataMat)
    # plot_data(dataMat)  # 画图
    # regTree = createTree(dataMat)
    # print(regTree)

    # dataMat2 = loadDataSet('./data/ex2.txt')
    # dataMat2 = np.mat(dataMat2)
    # # plot_data(dataMat2)  # 画图
    # mytree = createTree(dataMat2,ops=(0,1))
    # # print(mytree)
    # testMat = loadDataSet('./data/ex2test.txt')
    # testMat = np.mat(testMat)
    # prunedTree = prune(mytree,testMat)
    # print(prunedTree)

    # 模型树
    # dataMat2 = loadDataSet('./data/exp2.txt')
    # dataMat2 = np.mat(dataMat2)
    # plot_data(dataMat2)  # 画图
    # mytree = createTree(dataMat2, modelLeaf,modelErr,(1, 10))
    # print(mytree)

    # 基于人的智力水平和自行车的速度的关系
    trainMat = np.mat(loadDataSet('./data/bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('./data/bikeSpeedVsIq_test.txt'))
    print("-" * 20 + '回归树' + "-" * 20)
    tree = createTree(trainMat, ops=(1, 20))
    # print(tree)
    yHat = createForeCast(tree, testMat[:, 0])
    print("R平方为：",
          np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])  # 打印R平方，即相关系数

    print("-" * 20 + '模型树' + "-" * 20)
    tree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    # print(tree)
    yHat = createForeCast(tree, testMat[:, 0], modelTreeEval)
    print("R平方为：",
          np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])  # 打印R平方，即相关系数

    print("-" * 20 + '标准线性回归' + "-" * 20)
    ws, x, y = linearSolve(trainMat)
    print("回归系数为：\n", ws)
    for i in range(testMat.shape[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print("R平方为：",
          np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])  # 打印R平方，即相关系数
