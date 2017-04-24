from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

sc = SparkContext(appName="decisiontreeExample")
data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
(trainingData, testData) = data.randomSplit([0.7,0.3])

model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},impurity="gini", maxDepth=5, maxBins=32)

predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda p: p.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count() / float(testData.count())

print("Test Error: " + str(testErr))
print("Learned classification tree model:")
print(model.toDebugString())
model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
