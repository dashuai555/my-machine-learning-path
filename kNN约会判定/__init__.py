#!/usr/bin/python
# -*- coding:utf-8 -*-
import kNN
import numpy as np
import matplotlib.pyplot as plt
# group,labels=kNN.createDataSet()
# c=kNN.classify0([0,0],group,labels,3)
# print(c)


dataSet,labels=kNN.file2matrix('machine_learning/dataset.txt')
dataSet,ranges,minVals=kNN.autoNorm(dataSet)
##################################################################################################################


# x=[]
# y=[]
# for i in range(1,11):
#     ratio=0.05*i
#     print(ratio)
#     x.append(ratio)
#     errRatio=kNN.datingClassTest(ratio)
#     y.append(errRatio)


# fig=plt.figure(num=1)
# ax=fig.add_subplot(111)
# ax.plot(x,y,label='errRatio with given h0ratio')
# ax.scatter(x,y,10,c='b')
# plt.legend(loc='lower right')
# for i,j in zip(x,y):
#     plt.text(i+0.001,j+0.001,'%.3f' % j,ha='center',va='top')
# new_ticks1=np.arange(0,0.6,0.05)
# new_ticks2=np.arange(0,0.1,0.01)
# plt.xticks(new_ticks1)
# plt.yticks(new_ticks2)
# plt.show()
#############################################################################################################






# print(dataSet,ranges,minVals)

# type1_x=[]
# type1_y=[]
# type2_x=[]
# type2_y=[]
# type3_x=[]
# type3_y=[]
# # print ('range(len(labels)):',len(labels))
# # plt.scatter(dataSet[:,0],dataSet[:,1],15.0*np.array(labels),15.0*np.array(labels))
# for i in range(len(labels)):
#     if labels[i]==1:
#         type1_x.append(dataSet[i][1])
#         type1_y.append(dataSet[i][2])
#     if labels[i]==2:
#         type2_x.append(dataSet[i][1])
#         type2_y.append(dataSet[i][2])
#     if labels[i]==3:
#         type3_x.append(dataSet[i][1])
#         type3_y.append(dataSet[i][2])
# fig=plt.figure(num=1)
# ax=fig.add_subplot(111)

#以第二列和第三列为x,y轴画出散列点，给予不同的大小和颜色，第一个15.0*np.array(datingLabels)表示散列点的大小，
# 第二个15.0*np.array(datingLabels)表示散列点的颜色
#scatter（x,y,s=1,c="g",marker="s",linewidths=0）  
#s:散列点的大小,c:散列点的颜色，marker：形状，linewidths：边框宽度 


# # 把三种label提取出来
# axes2=plt.subplot(111)

# type1=ax.scatter(type1_x,type1_y,25,c='#357843')
# type2=ax.scatter(type2_x,type2_y,25,c='#ff9999')
# type3=ax.scatter(type3_x,type3_y,25,c='#99ffff')
# plt.xlabel('percentage of the time spent on playing games')
# plt.ylabel('Ice cream comsumed per week')
# ax.legend((type1,type2,type3),('dislike','smallDoses','largeDoses'),loc='best')


# plt.show()
###################################################################################################################
kNN.classifyPerson()