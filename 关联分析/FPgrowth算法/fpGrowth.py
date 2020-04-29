# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : fpGrowth.py
# Time       ：2020/3/23 19:12 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：
"""


class treeNode:
    """FP树的类定义"""

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 存放节点的名字
        self.count = numOccur  # 计数值
        self.nodeLink = None  # 用于链接相似的元素项
        self.parent = parentNode  # 指向当前节点的父节点
        self.children = {}  # 当前节点的子节点

    def inc(self, numOccur):
        """对count变量增加给定值"""
        self.count += numOccur

    def displayFPTree(self, ind=1):
        """将树以文本形式显示出来"""
        print("  " * ind, self.name, "  ", self.count)
        for child in self.children.values():
            child.displayFPTree(ind + 1)


# FP树构建函数
def createTree(dataSet, minsup=1):
    """构建FP树，其中输入的数据集dataSet是字典类型"""
    # 对数据集进行第一次扫描，统计每个元素项出现的频率，将结果存在头指针表中
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 去掉头指针表中出现次数小于最小支持度阈值的项
    for item in list(headerTable.keys()):
        if headerTable[item] < minsup:
            del (headerTable[item])
    freqItemSet = set(headerTable.keys())  # 获取频繁项
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0:
        return None, None

    # 对头指针表进行扩展，以便可以保存计数值和指向每种类型第一个元素项的指针
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]
    # 创建只包含空集合的根节点
    retTree = treeNode('Null Set', 1, None)
    # 对数据集进行第二次扫描，构建FP树
    for trans, count in dataSet.items():
        localD = {}
        # 对该项集进行过滤
        for item in trans:
            if item in freqItemSet:  # 仅考虑频繁项
                localD[item] = headerTable[item][0]
        # 对该项集进行排序，按元素的频率来排序,如果两元素在频次相同按字母顺序排序
        if len(localD) > 0:
            # ord() 函数是以一个字符（长度为1的字符串）作为参数，返回对应的十进制ASCII数值，比如：ord('a') 返回97  ord('b') 返回98
            # 如果都是数字 -ord(p[0]) 改为 int(p[0])
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: (
            p[1], -ord(p[0])), reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    """更新FP树，让FP树生长"""
    # 判断项集中第一个元素项是否作为子节点存在，如果存在，更新该元素项的计数
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:  # 不存在，则创建新的节点，并作为子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表，以指向新的节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeadera(headerTable[items[0]][1], inTree.children[items[0]])
    # 不断调用自身函数，每次调用会去掉列表中第一个元素
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeadera(nodeToTest, targetNode):
    """确保节点链接指向树中该元素项的每一个实例"""
    # 从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 从一棵FP树中挖掘频繁项集
# 发现以给定元素项结尾的所有路径函数
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)  # 迭代上溯整棵树
        if len(prefixPath) > 1:
            # 前缀路径对应的计数值
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


# 递归查找频繁项集的mineTree
def mineTree(inTree, headerTable, minsup, preFix, freqItemList):
    # 从头指针表的低端开始
    # headerTable.items():类似[('z', [5, None]），('r', [3,None]]格式，所以按p[1][0]排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)  # set类型的用add方法添加
        freqItemList.append(newFreqSet)
        # 从条件模式基中构建条件FP树
        condPattBase = findPrefixPath(headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBase, minsup)
        # 挖掘条件FP树
        if myHead != None:
            print('conditional tree for :', newFreqSet)
            myCondTree.displayFPTree()
            mineTree(myCondTree, myHead, minsup, newFreqSet, freqItemList)


# 简单数据和数据包装器
def loadData():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


if __name__ == '__main__':
    dataSet = loadData()
    initSet = createInitSet(dataSet)
    minsup=3
    myFPTree,myheaderTable = createTree(initSet, minsup)
    print(myheaderTable)
    print('构建的FP树：')
    myFPTree.displayFPTree()
    freqItems = []
    print("显示所有的条件树")
    mineTree(myFPTree,myheaderTable,minsup,set([]),freqItems)
    print('频繁项集：\n',freqItems)
