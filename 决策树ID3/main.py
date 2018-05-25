import trees
import tree
import treePlotter

fr=open('machine_learning/决策树/lenses.txt')
# 可以记一下，直接将文件分割为[[],[],....,[]]的list形式
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=tree.createTree(lenses,lensesLabels)
print(lensesLabels)
treePlotter.createPlot(lensesTree)
tree.storeTree(lensesTree,'machine_learning/决策树/theTree.txt')
theTree=tree.grabTree('machine_learning/决策树/theTree.txt')
print(theTree)
print(lensesLabels)
