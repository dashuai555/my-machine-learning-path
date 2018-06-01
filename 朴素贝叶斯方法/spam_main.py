import bayes
import matplotlib.pyplot as plt
import numpy as np
import feedparser
import Rss

sumErrRate=0.0
for i in range(100):
    sumErrRate+=bayes.spamTest()
aveErrRate=sumErrRate/100
print(aveErrRate)