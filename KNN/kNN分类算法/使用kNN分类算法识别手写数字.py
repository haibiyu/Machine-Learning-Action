# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 使用kNN分类算法识别手写数字.py
# Time       ：2020/3/28 19:50 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
from KNN.kNN分类算法.kNN分类算法 import kNN_classify
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2Vector(filename):
    """把一个32*32的二进制图像矩阵转换为1*1024的向量"""
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(lineStr[j])
    return returnVect


def handWritingClassTest_by_sklearn():
    """手写数字识别系统的测试代码"""
    hwLabels =[]
    traingFileList = os.listdir('trainingDigits')
    m = len(traingFileList)
    trainMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(classNumStr)
        trainMat[i,:] = img2Vector('trainingDigits/%s' %(fileNameStr))
    # 构建kNN分类器
    # metric：用于指定距离的度量指标，默认为闵可夫斯基距离
    # p：当参数metric为闵可夫斯基距离时，p=1，表示计算点之间的曼哈顿距离；p=2，表示计算点之间的欧氏距离；该参数的默认值为2；
    neigh = KNN(n_neighbors=3, algorithm='auto')  # 距离计算方法默认为欧氏距离
    # 拟合模型, trainMat为训练数据集,hwLabels为对应的标签集
    neigh.fit(trainMat, hwLabels)

    testFileList = os.listdir('testDigits')
    m_test = len(testFileList)
    error_count = 0.0
    for i in range(m_test):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testVec = img2Vector('testDigits/%s' % (fileNameStr))
        classifyResult = int(neigh.predict(testVec))
        # print("the classifier came back with: %d, the real answer is: %d" % (classifyResult,classNumStr))
        if classifyResult != classNumStr:
            error_count += 1.0
    print("\nthe total number of errors is: %d", error_count)
    print("\nthe total error rate is : %f" % (error_count/m_test))


def handWritingClassTest_by_theory():
    """手写数字识别系统的测试代码"""
    hwLabels =[]
    traingFileList = os.listdir('trainingDigits')
    m = len(traingFileList)
    trainMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(classNumStr)
        trainMat[i,:] = img2Vector('trainingDigits/%s' %(fileNameStr))
    testFileList = os.listdir('testDigits')
    m_test = len(testFileList)
    error_count = 0.0
    for i in range(m_test):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testVec = img2Vector('testDigits/%s' % (fileNameStr))
        classifyResult = int(kNN_classify(testVec,trainMat,hwLabels,3))
        # print("the classifier came back with: %d, the real answer is: %d" % (classifyResult,classNumStr))
        if classifyResult != classNumStr:
            error_count += 1.0
    print("\nthe total number of errors is: %d" % (error_count))
    print("\nthe total error rate is : %f" % (error_count/m_test))


if __name__ == '__main__':
    print("按kNN原理实现手写数字识别算法")
    start_time = time.time()
    handWritingClassTest_by_theory()
    print("\n耗时：",time.time()-start_time)

    print("-"*40)
    print("\n调用sklearn中的kNN算法实现手写数字识别算法")
    start_time = time.time()
    handWritingClassTest_by_sklearn()
    print("\n耗时：", time.time() - start_time)