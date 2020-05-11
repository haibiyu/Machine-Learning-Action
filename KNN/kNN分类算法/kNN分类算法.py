# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : kNN分类算法.py
# Time       ：2020/3/28 17:26 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
海伦收集的约会数据
包括三个特征：每年飞行常客里程数，玩视频游戏所耗时间百分比，每周消费的冰淇淋公升数
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def file2matrix(filename):
    """
    将文本记录转换为numpy的解析程序
    :param filename:文件路径名
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numOLines = len(arrayOLines)
    retMat = np.zeros((numOLines, 3))
    labels = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   # 截取掉所有的回车字符
        listFromLine = line.split('\t')  # 使用TAB字符\t进行分隔
        retMat[index, :] = listFromLine[:3]  # 前三列是特征值
        labels.append(int(listFromLine[-1]))  # 最后一列表示标签
        index += 1
    return retMat, labels

def autoNorm(dataSet):
    """对特征值进行最大-最小归一化，将数据范围处理到0~1之间
       计算公式：（x-min）/(max-min)"""
    minVals = dataSet.min(0)  # 从列中选出最小值
    maxVals = dataSet.max(0)
    rangeVals = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet-np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(rangeVals,(m,1))  # 特征值相除
    return normDataSet, rangeVals, minVals

def kNN_classify(X, dataSet, labels, k):
    """
    k近邻算法
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
    soretedDistIndex = distances.argsort()
    classCounts = {}
    for i in range(k):
        votelabel = labels[soretedDistIndex[i]]
        classCounts[votelabel] = classCounts.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCounts.items(), key=lambda p: p[1],
                              reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    """验证分类器，计算错误率"""
    hoRatio = 0.1  # 测试数据占总数据的比例
    # 海伦收集约会数据，前三列为特征，最后一列为标签
    dataSet, labels = file2matrix('datingTestSet2.txt')
    # 由于特征值之间的取值范围差异较大，故需对特征值进行归一化
    normDataSet, rangeVals, minVals = autoNorm(dataSet)
    m = normDataSet.shape[0]
    numTestSet = int(m*hoRatio)  # 测试数据量
    errorCount = 0.0
    # 训练数据
    trainX, trainLabel = normDataSet[numTestSet:, :], labels[numTestSet:]
    for i in range(numTestSet):
        clssifyResult = kNN_classify(normDataSet[i,:],trainX,trainLabel,3)
        # print("the classify came back with : %d, the real answer is: %d" %(clssifyResult,labels[i]))
        if clssifyResult != labels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/numTestSet))  # 错误率

def classifyPerson_by_theory():
    """约会网站预测函数，根据输入数据使用分类器获得预测分类结果"""
    # 讨厌、一般喜欢、非常喜欢
    resultList = ['not at all','in small doses','in large doses']
    ffMiles = float(input('frequent flier miles earned per year?'))
    percentTats = float(input('percentage of time spend playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,rangeVals,minVals = autoNorm(datingDataMat)
    X = [[ffMiles, percentTats, iceCream]]  # 数组大小（1,3）
    normX = (X-minVals)/rangeVals
    # 按kNN原理实现
    classifyResult = kNN_classify(normX,normMat,datingLabels,3)

    print('You will probably like this person : ',resultList[classifyResult-1])

def classifyPerson_by_sklearn():
    """约会网站预测函数，根据输入数据使用分类器获得预测分类结果"""
    # 讨厌、一般喜欢、非常喜欢
    resultList = ['not at all','in small doses','in large doses']
    ffMiles = float(input('frequent flier miles earned per year?'))
    percentTats = float(input('percentage of time spend playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,rangeVals,minVals = autoNorm(datingDataMat)
    X = [[ffMiles, percentTats, iceCream]]  # 数组大小（1,3）
    normX = (X - minVals) / rangeVals
    # 调用sklearn kNN包
    kNN_model = KNeighborsClassifier(n_neighbors=3,algorithm='auto') # 距离计算方法默认为欧氏距离
    kNN_model.fit(normMat,datingLabels)
    classifyResult = int(kNN_model.predict(normX))

    print('You will probably like this person : ',resultList[classifyResult-1])


if __name__ == '__main__':
    # 测试算法
    # datingClassTest()
    # 约会网站预测某一对象可交往程度
    # print("按kNN原理实现")
    # classifyPerson_by_theory()  # 按原理实现
    print("\n调用sklearn中的kNN算法实现")
    classifyPerson_by_sklearn()  # 调用sklearn包
