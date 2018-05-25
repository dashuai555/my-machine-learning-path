import matplotlib.pyplot as plt

decisionNode=dict(boxstyle='sawtooth',fc='0.8')
leafNode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # pyplot.annatate(s,xy, xytext=None, xycoords=’data’,textcoords=’data’, arrowprops=None, **kwargs)
    # 是matplotlib.pyplot模块提供的一个注解函数，可以用来对坐标中的数据进行注解，让人更清晰的得知坐标点得意义，
    # 现在对其参数作用进行分析：

    # xy -- 为点的坐标
    # xytext -- 为注解内容位置坐标，当该值为None时，注解内容放置在xy处
    # xycoords and textcoords 是坐标xy与xytext的说明，若textcoords=None，则默认textNone与xycoords相同，若都未设置，默认为data，
    # arrowprops -- 用于设置箭头的形状，类型为字典类型 arrowName = dict(arrowstyle="箭头的形状",connectionstyle="例如arc3")
    # **kwargs -- 用于接收其他设置参数，比如bbox用于设置文本的边框形状 类型也是dict 如bbox=dict（）
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,\
    textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
    
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=fig.add_subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    return

# 获得叶结点的数目和树的层数
def getNumLeafs(myTree):
    # 初始为零
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    # secondDict是一个dict而不是字符
    secondDict=myTree[firstStr]
    # 如果是dict类型则迭代，否则+1
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__=='dict':
            numLeafs=numLeafs+getNumLeafs(secondDict[key])
        else:
            numLeafs=numLeafs+1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth 
# 在父子节点之间添加文字注释
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,fontsize=8)
# 和getTreeDepth和getNumLeafs类似，也是个递归函数
def plotTree(myTree,parentPt,nodeTxt):
    # 计算宽度和深度
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    # 取当前特征的标签
    firstStr=list(myTree.keys())[0]
    # 计算初始子节点位置（其实就是第一个决策节点的位置即.5，1）
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    # 画中间的值，nodeTxt为特征值
    plotMidText(cntrPt,parentPt,nodeTxt)
    # 画出第一个决策节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

        
