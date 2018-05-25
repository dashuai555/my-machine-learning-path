import numpy as np
import matplotlib.pyplot as plt
import operator
# listdir可以列出给定目录的文件名
from os import listdir


# 将图像格式化处理为一个向量，这里将32*32处理成为1*1024
def img2vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

# 测试k-近邻算法识别手写数字
def handwritingClassTest():
    # hwLabels是测试集手写体的标签集
    hwLabels=[]
    trainingFileList=listdir('machine_learning/kNN手写识别系统/digits/trainingDigits')
    m=len(trainingFileList)
    # trainingMat为训练集
    trainingMat=np.zeros((m,1024))
    # 迭代m次
    for i in range(m):
        fileNameStr=trainingFileList[i]
        # 这句是为了去掉后缀txt
        fileStr=fileNameStr.split('.')[0]
        # 判别该文件是数字几的训练集
        classNumStr=int(fileStr.split('_')[0])
        # 将标签和训练数据导入标签集和数据集
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('machine_learning/kNN手写识别系统/digits/trainingDigits/%s'%fileNameStr)
    testFileList=listdir('machine_learning/kNN手写识别系统/digits/testDigits')
    errCount=0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('machine_learning/kNN手写识别系统/digits/testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with:%d,the real answer is %d'%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errCount=errCount+1
    print("\n the total number of errors is %d "%errCount)
    print("\n the total error rate is: %f"%(errCount/float(mTest)))


# 分类器0
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
 
    ##将inX有1*m的矩阵拓展成为n*m的矩阵并且逐项与dataSet中数据相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    
    ##将diffMat的每项都平方
    sqDiffMat = diffMat ** 2
   
    # 逐行将每行各项的各项相加求和得到一个##1*n##的矩阵,再将该矩阵的每个元素都开方
    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances ** 0.5


    # 将上面得到的距离列向量按从小到大排序,得到的array是第0位的元素在原array的位置，第1位的元素在原array的位置，以此类推
    sortedDistIndicies = distances.argsort()

    # 建立一个字典，对应分类在前k近的距离的样本中出现的次数
    classCount = {}
    # 迭代k次
    for i in range(k):
        # voteIlabel为第i近的标签名
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中该标签名的次数加一,之前没有出现过就给个默认值为0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sorted()函数产生一个新的列表,()内为(iterable迭代方式:这里面的列表可以直接写列表,但是dict要写.
    # items(),items()为返回可遍历的键/值; cmp:有默认值,可以选择由什么key决定排序方式;
    # key:用列表元素的某个属性和函数进行作为关键字，有默认值,这里面的key=operator.itemgetter(1)是
    # 定义了一个函数,意思是key为取第一项;reverse:是否采用反序)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    # 返回sortedClassCount列表的第0个的第0项
    return sortedClassCount[0][0]

