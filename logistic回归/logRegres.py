import numpy as np
import matplotlib.pyplot as plt

# 从文件中提取数据集和标签集
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('machine_learning/logistic回归/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid函数
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

# 批梯度下降
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    print(dataMatrix,labelMat)
    m,n=np.shape(dataMatrix)
    # alpha设置‘
    alpha=0.001
    maxCycle=500
    weights=np.ones((n,1))
    # 关键步骤 最大循环次数maxCycle可以自己设
    for k in range(maxCycle):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    m=np.shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range (m):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.linspace(-3.0,3.0,100)
    y=(-weights[1]*x-weights[0])/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0