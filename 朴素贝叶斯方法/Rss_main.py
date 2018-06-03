import Rss
import feedparser as fp
import re
import bayes
import numpy as np

sm=fp.parse('http://www.nba.com/cavaliers/rss.xml')
rj=fp.parse('https://www.nba.com/lakers/rss.xml')

sm_content=sm.entries
docList=[]
fullText=[]
for e in sm_content:
    print('title:',e.title)
    print('url:',e.link)
    print('content:',e.summary)
    doc=bayes.textParse(e.summary)
    docList.append(doc)
    fullText.extend(doc)

print (docList,'\n',fullText)
vocabList=set([])
for doc in docList:
    vocabList=vocabList|set(doc)
vocabList=list(vocabList)
print ('####################\nvocabList is :',vocabList)
trainMat=[]
for doc in docList:
    vec=np.zeros(len(vocabList))
    for word in doc:
        if word in vocabList:
            vec[vocabList.index(word)]+=1
        else:print ('the word:%s is not existed'%word)
    trainMat.append(vec)
print (trainMat)