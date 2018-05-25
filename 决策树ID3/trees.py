
from math import log
import operator
########################################################
# calcShannonEnt(dataSet)：利用所有类标签的发生频率计算出类别出现的概率，利用该概率计算香农熵
# createDataSet()：建立简单的鱼类鉴别数据集（用不到）
# splitDataSet(dataSet,axis,value)：按照给定特征划分数据集
# chooseBestFeatureToSplit(dataSet)：选择最好的数据集划分方式
# majorityCnt(classList):选择具有最多数目的类,classList是类别集，即每个样本的类别
# createTree(dataSet,labels):创建树的函数代码,dataSet数据集，labels是数据集中样本的特征的标签集，而不是分类结果的标签集
# storeTree(inputTree,filename)、grabTree(filename)：储存树和获取树
# classify(inputTree,featLabels,testVec):使用决策树的分类函数 
#########################################################
# 利用所有类标签的发生频率计算出类别出现的概率，利用该概率计算香农熵。 
def calcShannonEnt(dataSet):
    # numEntriers-dataSet中training examples的总数
    numEntries=len(dataSet)
    labelCounts={}
    # 对于每一个数据集中的example，把分类提取出来，统计进之前建立好的dict中
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1   
    shannonEnt=0.0
    # 计算香农熵=-Σp(x)*log2p(x)______熵越高，则混合的数据越多
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt=shannonEnt-prob*log(prob,2)
    return shannonEnt

# 建立简单的鱼类鉴别数据集
def createDataSet():
    dataSet=[[1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,1,'no'],
    [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

# 按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        # 抽取符合条件的特征向量，
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            # extend和append不同，要注意，extend不改变列表格式
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # numFeatures-数据集中example的特征数量
    numFeatures=len(dataSet[0])-1
    # 计算原始的熵值
    baseEntropy=calcShannonEnt(dataSet)
    # 设置最大信息增益和最大信息增益对应的特征index
    bestInfoGain=0.0
    bestFeature=-1
    # 对于特征中的每个特征
    for i in range (numFeatures):
        # 建立特征列表，列表中是每个样本的第i个特征值
        featList=[example[i] for example in dataSet]
        # 将featList中的值去重
        uniqueVals=set(featList)
        # 第i个特征的熵值为newEntropy
        newEntropy=0.0
        # 对于第i个特征的每个特征值，提取矩阵，并且计算提取之后的熵
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            # 提取之后的熵=Σp(value)*熵(value)
            newEntropy=newEntropy+prob*calcShannonEnt(subDataSet)
        # 信息增益=初始熵-第i个特征提取后的熵，越大证明信息增益越高
        infoGain=baseEntropy-newEntropy
        # 如果是最大信息增益的话，改变最佳特征值
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
        return bestFeature

# 递归构建决策树，递归结束的条件：遍历完所有划分数据集的属性or每个分支下的所有实例都具有相同的分类

# 选择具有最多数目的类 
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        classCount[vote]=classCount.get(vote,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 使用决策树的分类函数 
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict:
        if key==testVec[featIndex]:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

# 创建树的函数代码,dataSet数据集，labels是数据集中样本的特征的标签集，而不是分类结果的标签集
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    # 若是类别完全相同则退出
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 若遍历完所有特征时，返回出现次数最多的类别-用majorityCnt
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    # 将bestFeat从labels中删除
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    
    return myTree

# 储存树和获取树
def storeTree(inputTree,filename):
    import pickle
    # 必须是'wb','w'会报错
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    # 同上面必须是'rb'
    fr=open(filename,'rb')
    return pickle.load(fr)