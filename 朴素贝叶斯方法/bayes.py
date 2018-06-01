import numpy as np


# 就是构建了个dataSet和classVec，没啥的
def loadingDataSet():
    postingList=[['my','dog','has','flea','problems','help','please']
    ,['maybe','not','take','him','to','dog','park','stupid']
    ,['my','dalmation','is','so','cute','I','love','him']
    ,['stop','posting','stupid','worthless','garbage']
    ,['mr','licks','ate','my','steak','how','to','stop','him']
    ,['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

# 建立词汇列表vocabSet
def createVocabList(dataSet):
    # 创建一个空集
    vocabSet=set([])
    # 对每个样本与之前建立的集取并
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

# 建立对应每个词的returnVec：[0,1,1,0...]0为未出现在信件中，1为出现了，输入为词汇表和某个文档，输出为文档向量
# 朴素贝叶斯词集模型
def setOfWords2Vec(vocabList,inputSet):
    # 建立一个长为vocablist的向量
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print ('the word: %s is not in my Vocabulary!'%word)
    return returnVec

# 朴素贝叶斯词袋模型
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print ('the word: %s is not in my Vocabulary!'%word)
    return returnVec

# 朴素贝叶斯分类器训练函数
####################################################
# 计算每个类别的文档数目m1,m2,m3...
# 对每篇训练文档：
#   对每个类别：
#       如果词条出现在文档中->增加该词条的计数值
#       增加所有词条的计数值
# 对每个类别：
#   对每个词条：
#       将该词条的数目除以词条数目得到条件概率
# 返回每个类别的条件概率
####################################################
# 朴素贝叶斯分类器训练函数，trainMatrix：文档矩阵；trainCategory：文档类别标签向量
# 标签向量中1代表为垃圾邮件，0代表为非垃圾邮件
def trainNB(trainMatrix,trainCategory):
    # 该分类训练文档的数目
    numTrainDocs=len(trainMatrix)
    # 词典中词语的数目
    numWords=len(trainMatrix[0])
    # 初始化概率,pAbusive是邮件为垃圾邮件的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    # p0Num用来记录所有0类的单词各出现的次数
    # p0Denom用来记录0类单词出现的总次数
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=numWords
    p1Denom=numWords
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    # p1Vect是类别为一时，各个词语出现的概率,取对数以防止数太小而导致无法正常工作
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
# 朴素贝叶斯分类器
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # ln(p(x1|y=1)p(x2|y=1)....p(xn|y=1)p(y=1))=ln(p(x1|y=1))+ln(p(x2|y=1))+...+ln(p(y=1))
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
# 便利函数，只是将之前的代码综合起来
def testingNB():
    listOPosts,listClasses=loadingDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postInDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postInDoc))
    p0V,p1V,pAb=trainNB(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#  文件解析
def textParse(bigString):
    import re
    # 用除字母和数字之外的字符分隔邮件
    listOfTokens=re.split(r'\W*',bigString)
    # 只要长度大于2的分割单元
    return [tok.lower() for tok in listOfTokens if len(tok)>2 ]

# 垃圾邮件测试函数
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    # 导入并解析文件
    # 将每封邮件分片并且加入docList成为列表
    # 
    for i in range(1,26):
        wordList=textParse(open('朴素贝叶斯方法/email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('朴素贝叶斯方法/email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=[]
    for i in range(50):
        trainingSet.append(i)
    print(trainingSet)
    testSet=[]
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    print(trainingSet)
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    errorRate=float(errorCount)/len(testSet)
    return errorRate        
