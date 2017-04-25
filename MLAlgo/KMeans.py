from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext

sc = SparkContext(appName="KMeansExample")

data = sc.textFile("data/mllib/kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")
print(parsedData.collect())
def error(point):
	center = clusters.centers[clusters.predict(point)]
	return sqrt(sum([x**2 for x in (point - center)]))
test = parsedData.map(lambda point: clusters.predict(point))
print(test.collect())

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x,y: x+y)
print("Within Set Sum of Squared error = " + str(WSSSE))



