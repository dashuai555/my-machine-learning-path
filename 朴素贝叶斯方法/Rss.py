import feedparser as fp
from operator import itemgetter

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

def calcMostFreq(vocabList,fullText):
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=itemgetter(1),reverse=True)
    return sortedFreq[:30]