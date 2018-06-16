import logRegres

dataMat,labelMat=logRegres.loadDataSet()
print(dataMat)
print(labelMat)
weights=logRegres.gradAscent(dataMat,labelMat)
print(weights)
# matrix.getA()就是将mat转为Array
logRegres.plotBestFit(weights.getA())