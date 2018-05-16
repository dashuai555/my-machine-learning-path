#!/usr/bin/python
# -*- coding:utf-8 -*-
import kNN

group,labels=kNN.createDataSet()
c=kNN.classify0([0,0],group,labels,3)
print(c)