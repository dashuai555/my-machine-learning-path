#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    print(dataSetSize)
    ##将inX有1*m的矩阵拓展成为n*m的矩阵并且逐项与dataSet中数据相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    print(diffMat)
    ##将diffMat的每项都平方
    sqDiffMat = diffMat ** 2
    print(sqDiffMat)
    # 逐行将每行各项的各项相加求和得到一个##1*n##的矩阵,再将该矩阵的每个元素都开方
    sqDistances = sqDiffMat.sum(axis=1)
    print(sqDistances)
    distances = sqDistances ** 0.5
    print(distances)

    # 将上面得到的距离列向量按从小到大排序,得到的array是第0位的元素在原array的位置，第1位的元素在原array的位置，以此类推
    sortedDistIndicies = distances.argsort()
    print(sortedDistIndicies)
    # 建立一个字典，对应分类在前k近的距离的样本中出现的次数
    classCount = {}
    # 迭代k次
    for i in range(k):
        # voteIlabel为第i近的标签名
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中该标签名的次数加一,之前没有出现过就给个默认值为0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #sorted()函数产生一个新的列表,()内为(iterable迭代方式:这里面的列表可以直接写列表,但是dict要写.items(),items()为返回可遍历的键/值;
    # cmp:有默认值,可以选择由什么key决定排序方式;key:用列表元素的某个属性和函数进行作为关键字，有默认值,这里面的key=operator.itemgetter
    # (1)是定义了一个函数,意思是key为取第一项;reverse:是否采用反序)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    # 返回sortedClassCount列表的第0个的第0项
    return sortedClassCount[0][0]
