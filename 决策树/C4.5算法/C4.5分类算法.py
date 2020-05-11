# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : C4.5分类算法.py
# Time       ：2020/4/29 21:51 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
from math import log
import operator
import numpy as np
from 机器学习实战算法.决策树.ID3算法.ID3算法 import createPlot

def createDataSet():
    """构建数据集"""
    dataSet = [['是', '单身', 125, '否'],
               ['否', '已婚', 100, '否'],
               ['否', '单身', 70, '否'],
               ['是', '已婚', 120, '否'],
               ['否', '离异', 95, '是'],
               ['否', '已婚', 60, '否'],
               ['是', '离异', 220, '否'],
               ['否', '单身', 85, '是'],
               ['否', '已婚', 75, '否'],
               ['否', '单身', 90, '是']]
    labels = ['是否有房', '婚姻状况', '年收入(k)']  # 三个特征
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:给定的数据集
    :return:返回香农熵
    """
    numEntries = len(dataSet)
    labelCounts ={}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for label in labelCounts.keys():
        prob = float(labelCounts[label])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def majorityCnt(classList):
    """获取出现次数最好的分类名称"""
    classCount = {}
    classList= np.mat(classList).flatten().A.tolist()[0]  # 数据为[['否'], ['是'], ['是']], 转换后为['否', '是', '是']
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet,axis,value):
    """对离散型特征划分数据集"""
    retDataSet = []  # 创建新的list对象，作为返回的数据
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 抽取
            retDataSet.append(reducedFeatVec)
    return retDataSet


def splitContinuousDataSet(dataSet, axis, value, direction):
    """对连续型特征划分数据集"""
    subDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:  # 按照大于(>)该值进行划分
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
        if direction == 1:
            if featVec[axis] <= value:  # 按照小于等于(<=)该值进行划分
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
    return subDataSet

def chooseBestFeatureToSplit(dataSet, labels):
    """选择最好的数据集划分方式"""
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = 0
    baseGainRatio = -1
    numFeatures = len(dataSet[0]) - 1
    # 建立一个字典，用来存储每一个连续型特征所对应最佳切分点的具体值
    bestSplitDic = {}
    # print('dataSet[0]:' + str(dataSet[0]))
    for i in range(numFeatures):
        # 获取第i个特征的特征值
        featVals = [example[i] for example in dataSet]
        # 如果该特征时连续型数据
        if type(featVals[0]).__name__ == 'float' or type(
                featVals[0]).__name__ == 'int':
            # 将该特征的所有值按从小到大顺序排序
            sortedFeatVals = sorted(featVals)
            # 取相邻两样本值的平均数做划分点，共有 len(featVals)-1 个
            splitList = []
            for j in range(len(featVals) - 1):
                splitList.append(
                    (sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0)
            # 遍历每一个切分点
            for j in range(len(splitList)):
                # 计算该划分方式的条件信息熵newEntropy
                newEntropy = 0.0
                value = splitList[j]
                # 将数据集划分为两个子集
                greaterSubDataSet = splitContinuousDataSet(dataSet, i, value, 0)
                smallSubDataSet = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(greaterSubDataSet) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(greaterSubDataSet)
                prob1 = len(smallSubDataSet) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(smallSubDataSet)
                # 计算该划分方式的分裂信息
                splitInfo = 0.0
                splitInfo -= prob0 * log(prob0, 2)
                splitInfo -= prob1 * log(prob1, 2)
                # 计算信息增益率 = 信息增益 / 该划分方式的分裂信息
                gainRatio = float(baseEntropy - newEntropy) / splitInfo
                if gainRatio > baseGainRatio:
                    baseGainRatio = gainRatio
                    bestSplit = j
                    bestFeature = i
            bestSplitDic[labels[i]] = splitList[bestSplit]  # 最佳切分点
        else:  # 如果该特征时连续型数据
            uniqueVals = set(featVals)
            splitInfo = 0.0
            # 计算每种划分方式的条件信息熵newEntropy
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                splitInfo -= prob * log(prob, 2)  # 计算分裂信息
                newEntropy += prob * calcShannonEnt(subDataSet)  # 计算条件信息熵
            # 若该特征的特征值都相同，说明信息增益和分裂信息都为0，则跳过该特征
            if splitInfo == 0.0:
                continue
            # 计算信息增益率 = 信息增益 / 该划分方式的分裂信息
            gainRatio = float(baseEntropy - newEntropy) / splitInfo
            if gainRatio > baseGainRatio:
                bestFeature = i
                baseGainRatio = gainRatio
    # 如果最佳切分特征是连续型，则最佳切分点为具体的切分值
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(
            dataSet[0][bestFeature]).__name__ == 'int':
        bestFeatValue = bestSplitDic[labels[bestFeature]]
    # 如果最佳切分特征时离散型，则最佳切分点为 切分特征名称,【其实对于离散型特征这个值没有用】
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        bestFeatValue = labels[bestFeature]
    # print('bestFeature:' + str(labels[bestFeature]) + ', bestFeatValue:' + str(bestFeatValue))
    return bestFeature, bestFeatValue


def createTree(dataSet, labels):
    """创建C4.5树"""
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeature, bestFeatValue = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:bestFeature]
    subLabels.extend(labels[bestFeature + 1:])
    # 针对最佳切分特征是离散型
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        featVals = [example[bestFeature] for example in dataSet]
        uniqueVals = set(featVals)
        for value in uniqueVals:
            reduceDataSet = splitDataSet(dataSet, bestFeature, value)
            # print('reduceDataSet:' + str(reduceDataSet))
            myTree[bestFeatLabel][value] = createTree(reduceDataSet, subLabels)
            # print(myTree[bestFeatLabel][value])
    # 针对最佳切分特征是连续型
    if type(dataSet[0][bestFeature]).__name__ == 'int' or type(
            dataSet[0][bestFeature]).__name__ == 'float':
        # 将数据集划分为两个子集，针对每个子集分别建树
        value = bestFeatValue
        greaterSubDataSet = splitContinuousDataSet(dataSet, bestFeature, value, 0)
        smallSubDataSet = splitContinuousDataSet(dataSet, bestFeature, value, 1)
        # print('greaterDataset:' + str(greaterSubDataSet))
        # print('smallerDataSet:' + str(smallSubDataSet))
        # 针对连续型特征，在生成决策的模块，修改划分点的标签，如“> x.xxx”，"<= x.xxx"
        myTree[bestFeatLabel]['>' + str(value)] = createTree(greaterSubDataSet,subLabels)
        myTree[bestFeatLabel]['<=' + str(value)] = createTree(smallSubDataSet,subLabels)
    return myTree


if __name__ == '__main__':
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    dataSet, labels = createDataSet()
    mytree = createTree(dataSet, labels)
    print("最终构建的C4.5分类树为：\n",mytree)
    createPlot(mytree, decisionNode, leafNode)  # 绘制树


