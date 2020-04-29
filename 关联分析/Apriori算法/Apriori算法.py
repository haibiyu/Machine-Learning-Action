# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : Apriori算法.py
# Time       ：2020/3/19 9:24 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：关联规则--Apriori 算法
                优点:易编码
                缺点：在大数据集上可能较慢
"""


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 发现频繁项集
def createC1(dataSet):
    """
    创建候选项集C1
    :param dataSet:数据集（未集合化）
    :return:返回候选项集C1
    """
    C1 = []
    for transaction in dataSet:  # 每条清单
        for item in transaction:  # 清单中购买的每个物品
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # 对C1中的每个项构建一个不变子集


def scanD(D, Ck, minSupport):
    """
    扫描数据D，从 Ck 到 Lk，丢掉不满足最小支持度要求的项目
    :param D: 数据集（已经集合化）
    :param Ck: Ck
    :param minSupport: 最小支持度
    :return: 满足支持度的项集，以及该项集对应的支持度
    """
    ssCnt = {}
    for tid in D:
        for c in Ck:
            if c.issubset(tid):  # 如果集合c在该条清单中出现
                # 增加字典ssCnt中对应的计数值
                if c not in ssCnt:
                    ssCnt[c] = 1
                else:
                    ssCnt[c] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}  # 所有候选项集的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 计算各项的支持度
        if support >= minSupport:
            retList.insert(0, key)  # 将满足支持度的项插入到剩余项列表中
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    创建由k+1项组成的候选项集C(k+1)，由 频繁项集Lk 到 C（k+1）
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-2个项相同时，将两个集合合并
            # 比如L2:{0,1},{0,2},k-2就是0,即如果第一个元素相同，则进行合并
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    """
    获取所有满足条件的频繁项集的列表，和所有候选项集的支持度信息
    :param dataSet: 数据集（未集合化）
    :param minSupport:最小支持度
    :return:
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet)) # 将dataSet集合化，以满足scanD的格式要求
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)  # 创建Ck
        # 扫描数据，从 Ck 到 Lk，丢掉不满足最小支持度要求的项
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK) # 将新的项集的支持度数据加入原来的总支持度字典中
        L.append(Lk)
        k += 1
    return L, supportData

# 从频繁项集中挖掘关联规则
def generateRules(L, supportData, minConf=0.7):
    """
    根据频繁项集和最小可信度生成规则集合
    :param L:频繁项集列表
    :param supportData:存储着所有项集（不仅仅是频繁项集）的支持度，是字典类型
    :param minConf:最小可信度阈值
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):  # 只获取有两个或更多元素的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  # 创建只包含单个元素集合的列表
            if i > 1:  # 如果频繁项集中的元素个数大于2，需要进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 如果频繁项集中只有两个元素，则用calConf（）
                calConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    规则生成与评估，计算规则的可信度以及找到满足最小可信度要求的规则
    :param freqSet:频繁项集
    :param H:频繁项集中所有的元素
    :param supportData:频繁项集中所有元素的支持度，是字典类型
    :param brl:满足可信度条件的关联规则列表
    :param minConf:最小可信度阈值
    :return:
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, "--->", conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    从最初的项集中生成更多的关联规则，对频繁项集中元素超过2的项集进行合并
    :param freqSet:频繁项集
    :param H: 频繁项集中的所有元素，即出现在规则右部的元素列表
    :param supportData: 频繁项集中所有元素的支持度，是字典类型
    :param brl: 满足可信度条件的关联规则列表
    :param minConf: 最小可信度阈值
    :return:
    """
    m = len(H[0])  # H中频繁项集的大小
    # 查看频繁项集是否大到移除大小为 m　的子集，尝试进一步合并
    if len(freqSet) > (m + 1):
        Hmpl = aprioriGen(H, m + 1)  # 创建Hm+1条新候选规则
        Hmpl = calConf(freqSet, Hmpl, supportData, brl, minConf)
        # # 如果不止一条规则满足要求，则需要进一步递归合并，组合这些规则
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)


if __name__ == '__main__':
    dataSet = loadDataSet()
    # print(dataSet)
    # C1 = createC1(dataSet)
    # print("C1:\n",C1)
    # D = list(map(set, dataSet))
    # print("D:\n",D)
    # L1,supportData = scanD(D,C1,0.5)
    # print("L1:\n",L1)
    # print("supportData:\n",supportData)

    L, supportData = apriori(dataSet, 0.5)
    print("频繁项集L:\n", L)
    print("所有候选项集的支持度信息:\n", supportData)
    print("关联规则：")
    rules = generateRules(L, supportData, minConf=0.7)
