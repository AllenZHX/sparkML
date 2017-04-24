from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
	conf = SparkConf().setMaster("local[4]").setAppName("RandomForestExample")
	sc = SparkContext(conf=conf)
	data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	(trainingData, testData) = data.randomSplit([0.7,0.3])

	model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=3, featureSubsetStrategy="auto", impurity="gini", maxDepth=4, maxBins=32)

	predictions = model.predict(testData.map(lambda p: p.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testErr = 1.0 * labelsAndPredictions.filter(lambda (x, v): x != v).count() / testData.count()

	print("TestErro = " + str(testErr))
	print("Learned classification forest model:")
	print(model.toDebugString())

	model.save(sc, "target/tmp/myRandomForestClassificationModel")
	sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
