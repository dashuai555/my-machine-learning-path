import kNN1
import numpy as np 
import matplotlib.pyplot as plt 

dataSet,labels=kNN1.file2matrix('machine_learning/dataset.txt')
fig=plt.figure(num=0,figsize=((6,4)))
ax=fig.add_subplot(111)
type1_x=[]
type1_y=[]
type2_x=[]
type2_y=[]
type3_x=[]
type3_y=[]
for i in range(len(labels)):
    if labels[i]==1:
        type1_x.append(dataSet[i,1])
        type1_y.append(dataSet[i,2])
    if labels[i]==2:
        type2_x.append(dataSet[i,1])
        type2_y.append(dataSet[i,2])
    if labels[i]==3:
        type3_x.append(dataSet[i,1])
        type3_y.append(dataSet[i,2])
type1=ax.scatter(type1_x,type1_y,s=15,c='#cd4523')
type2=ax.scatter(type2_x,type2_y,s=15,c='#775d23')
type3=ax.scatter(type3_x,type3_y,s=15,c='#89bb23')
plt.legend((type1,type2,type3),('largeDoes','smallDoes','didntLikes'))
plt.show()
print('hello world')
