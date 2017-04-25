from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

def parsePoint(line):
	values = [float(x) for x in line.split(" ")]
	return LabeledPoint(values[0], values[1:])

sc = SparkContext(appName="PythonSVMExample")
#data = sc.textFile("data/mllib/sample_svm_data.txt")
data = sc.textFile("MLData/data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

model = SVMWithSGD.train(parsedData, iterations=10000)

labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v,p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

#
model.save(sc, "target/tmp/pythonSVMWithSGDModel")
#sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")
