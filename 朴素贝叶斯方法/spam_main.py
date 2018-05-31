import bayes
import matplotlib.pyplot as plt
import numpy as np

sum=0.0
for i in range(10):
    sum+=bayes.spamTest()
print('the avearge error rate is: %f'%(sum/10))