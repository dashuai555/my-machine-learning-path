import feedparser as fp
from operator import itemgetter
import numpy as np
import bayes

def test(url):
    # 学习使用feedparser
    # 输入url
    # 输出：页面信息
    one_page_dict=fp.parse(url)
    # 解析得到的是字典
    # 输出字典中的键值有哪些，一共有10中如下： 
    # ['feed', 'status', 'version', 'encoding', 'bozo', 'headers', 'href', 'namespaces', 'entries', 'bozo_exception'] 
    print(one_page_dict)
    print(one_page_dict['entries'])
    print(len(one_page_dict['entries']))

# 提取使用频率最高的30个词
def calcMostFreq(vocabList,fullText):
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=itemgetter(1),reverse=True)
    return sortedFreq[:30]

# Rss源分类器
def localWords(feed1,feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=bayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=bayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=bayes.createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=[]
    for i in range(2*minLen):
        trainingSet.append(i)
    testSet=[]
    for i in range(20):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=bayes.trainNB(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bayes.bagOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

        