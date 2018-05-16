#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

def computeCost(X,y,theta):
    inner=pow(np.dot(X,theta)-y,2)
    return 1/(2*len(X))*inner