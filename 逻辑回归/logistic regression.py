# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : logistic regression.py
# Time       ：2020/4/15 8:25 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import numpy as np

def loadDataSet():
    data=[]
    labels=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        currLine = line.strip().split()
        data.append([1.0,float(currLine[0]),float(currLine[1])])
        labels.append(int(currLine[2]))
    return data,labels

def sigmoid(inX):
    """返回sigmoid后的值"""
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataSet,labels):
    """用最优化算法（梯度上升）获取最优回归系数"""
    dataMat = np.mat(dataSet)
    labelsMat = np.mat(labels).transpose()  # 将列表转换为列向量，即 n*1
    m,n = dataMat.shape
    numIters = 500
    alpha = 0.01
    weights = np.ones((n,1))
    for i in range(numIters):
        h = sigmoid(dataMat*weights)
        error = labelsMat-h
        weights = weights+alpha*dataMat.transpose()*error  # w= w + alpha*(y-predict)*data
    return weights

def plotBestFit(weights):
    """画出数据集和逻辑回归最佳拟合直线"""
    import matplotlib.pyplot as plt
    dataSet,labels = loadDataSet()
    dataMat = np.mat(dataSet)
    m = dataMat.shape[0]
    xcoord1=[]
    ycoord1=[]
    xcoord2=[]
    ycoord2=[]
    for i in range(m):
        if int(labels[i]) == 1:
            xcoord1.append(dataMat[i,1])
            ycoord1.append(dataMat[i,2])
        else:
            xcoord2.append(dataMat[i,1])
            ycoord2.append(dataMat[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord1,ycoord1,s=30,c='red',marker='s')
    ax.scatter(xcoord2,ycoord2,s=30,c='green',marker='o')
    x=np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMat,labels):
    """用最优化算法（随机梯度上升）获取最优回归系数"""
    m,n = dataMat.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMat[i]*weights))
        error = labels[i]-h
        weights = weights+alpha*error*dataMat[i]
    return weights

def stocGradAscent1(dataMat,labels,numIters=150):
    """用最优化算法（改进的随机梯度上升）获取最优回归系数"""
    m, n = dataMat.shape
    weights = np.ones(n)
    for j in range(numIters):
        dataIndex=list(np.arange(m))
        for i in range(m):
            alpha = 4.0/(j+i+1.0) + 0.01  # 每次迭代时需要调整alpha,缓解数据波动
            # 随机选取样本来更新回归系数
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del dataIndex[randIndex]

    return weights

def classifyVector(inX,weights):
    """判断给定inX的分类"""
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0  # 表示未能存活
    else:
        return 0.0   # 表示仍存活


def colicTest():
    """导入训练和测试集，进行逻辑回归，计算测试集分类错误率"""
    frTrain = open('horseColicTrain.txt')
    frTest = open('horseColicTest.txt')
    # 获取训练集输入数据和标签
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine = line.strip().split(' ')
        lineArr=[]
        del currLine[2]  # 删除数据第三列，即医院登记号
        for i in range(21):  # 删除后数据第22列表示马活着、死掉或者安乐死
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # 对安乐死和死归类于死亡（这里是正例，用1表示；存活认为是反例用0表示）
        if int(float(currLine[21])) ==2 or int(float(currLine[21])) ==3:
            trainingLabels.append(1.0)
        else:
            trainingLabels.append(0.0)
    # 对训练集数据调用随机梯度上升算法，得到最优回归系数
    trainingweights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    # 获取测试集，并计算分类错误数
    errorCount=0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine=line.strip().split(' ')
        lineArr=[]
        del currLine[2]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(float(currLine[21])) ==2 or int(float(currLine[21])) ==3:
            curlabel=1.0
        else:
            curlabel=0.0
        if int(classifyVector(np.array(lineArr),trainingweights)) != int(curlabel):
            errorCount += 1
    # 计算测试集的分类错误率
    errorRate = float(errorCount)/numTestVec
    print('the error rate of this test is: %f' % (errorRate))
    return errorRate

def multiTest():
    """计算迭代10次的平均错误率"""
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTest,errorSum/float(numTest)))

if __name__ == '__main__':
    dataSet, labels = loadDataSet()
    weights = stocGradAscent1(np.array(dataSet), labels)
    plotBestFit(np.array(weights))
    # colicTest()
    # multiTest()