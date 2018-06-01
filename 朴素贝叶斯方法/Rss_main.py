import Rss
import feedparser as fp

ny=fp.parse('https://www.baidu.com/')
print(ny['entries'])
print(len(ny['entries']))