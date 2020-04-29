# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 从新闻网站点击流中挖掘.py
# Time       ：2020/3/25 21:30 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：
"""
from 关联分析.FPgrowth算法.fpGrowth import *

def createTree(dataSet,minsup=1):
    """构建FP树，其中输入的数据集dataSet是字典类型"""
    # 对数据集进行第一次扫描，统计每个元素项出现的频率，将结果存在头指针表中
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    # 去掉头指针表中出现次数小于最小支持度阈值的项
    for item in list(headerTable.keys()):
        if headerTable[item] < minsup:
            del (headerTable[item])
    freqItemSet = set(headerTable.keys())  # 获取频繁项
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0:
        return None,None

    # 对头指针表进行扩展，以便可以保存计数值和指向每种类型第一个元素项的指针
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]
    # 创建只包含空集合的根节点
    retTree = treeNode('Null Set', 1, None)
    # 对数据集进行第二次扫描，构建FP树
    for trans,count in dataSet.items():
        localD = {}
        # 对该项集进行过滤
        for item in trans:
            if item in freqItemSet: # 仅考虑频繁项
                localD[item] = headerTable[item][0]
        # 对该项集进行排序，按元素的频率来排序,如果两元素在频次相同按字母顺序排序
        if len(localD) > 0:
            # ord() 函数是以一个字符（长度为1的字符串）作为参数，返回对应的十进制ASCII数值，比如：ord('a') 返回97  ord('b') 返回98
            # 如果都是数字 -ord(p[0]) 改为 int(p[0])
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: (p[1], int(p[0])), reverse=True)]
            # print(orderedItems)
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

# 将数据集导入到列表
parsedDat = [line.split()  for line in open('kosarak.txt').readlines()]
# 对初始集合格式化
initSet = createInitSet(parsedDat)
# 构建FP树，找到那些至少呗5000人浏览过的新闻报道
myFPTree, myHeaderTable = createTree(initSet,5000)
print('构建的FP树：')
myFPTree.displayFPTree()
# 创建一个空列表来保存频繁项集
myFreqList = []
print("显示所有的条件树")
mineTree(myFPTree,myHeaderTable,5000,set([]),myFreqList)
print('以下这些新闻报道或报道集合曾经被5000人或者更多人浏览过')
for item in myFreqList:
    print(item)