import numpy as np 
import matplotlib.pyplot as pyplot
 

def file2matrix(filename):
    label={'largeDoses':1,'smallDoses':2,'didntLike':3}
    fr=open(filename)
    arrayOLines=fr.readlines()
    m=len(arrayOLines)
    dataSet=np.zeros((m,3))
    labels=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        listFromLine[-1]=label[listFromLine[-1]]
        dataSet[index,:]=listFromLine[:-1]
        labels.append(listFromLine[-1])
        index=index+1
    return dataSet,labels

    

