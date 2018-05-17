#!/usr/bin/python
# -*- coding:utf-8 -*-
import kNN

# group,labels=kNN.createDataSet()
# c=kNN.classify0([0,0],group,labels,3)
# print(c)
dataSet,labels=kNN.file2matrix('C:/Users/12076/PycharmProjects/practice/venv/machine_learning/dataset.txt')
print(dataSet)
print(labels)