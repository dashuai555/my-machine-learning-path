import Rss
import feedparser as fp

dict_1={'a':{'v':1,'h':2}}
print(dict_1['a'])
ny=fp.parse('http://feed.cnblogs.com/blog/u/161528/rss')
for e in ny.entries:
    print('title:',e.title)
    print('url:',e.id)
    # e.content是一个只有一个词典元素的列表，所以用e.content[0]将其变为词典
    print('content:',e.content[0].value)