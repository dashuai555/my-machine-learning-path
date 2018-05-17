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

    #############k-近邻算法改进约会网站的配对效果
    # 从文本中解析数据

def file2matrix(filename):
    labels={'largeDoses':2,'smallDoses':1,'didntLike':0}
    # 打开文件
    fr=open(filename)
    # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
    # 如果碰到结束符 EOF 则返回空字符串。
    # 返回列表，包含所有的行。
    # arrayOLines是一个n*1的向量
    arrayOLines=fr.readlines()
    # numberOfLines为n
    numberOfLines=len(arrayOLines)
    # np.zero()为返回来一个给定形状和类型的用0填充的数组，下面代码的意思就是创建一个n*3的全零矩阵
    returnMat=np.zeros((numberOfLines,3))
    # 创建一个标签向量
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        # strip()用于移除字符串头尾指定的字符（默认为空格），这一步应该是为了清洗数据,截取掉所有的回车字符
        line=line.strip()
        # split('char',num) 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
        # 这一句将每行字符串分割为一个1*4的列表
        listFromLine=line.split('\t')
        listFromLine[-1]=labels[listFromLine[-1]]
        # 将returnMat的每一行赋予listFromLine的前三个值，classLabelVector添加第四个值
        returnMat[index,:]=listFromLine[:3]
        classLabelVector.append(np.int(listFromLine[-1]))
        # index+1
        index=index+1
    return returnMat,classLabelVector





