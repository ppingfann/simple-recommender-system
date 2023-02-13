from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def oneHotEncoderExample(booksSamples):
    samplesWithIdNumber = booksSamples.withColumn("bookIdNumber", F.col("Year-Of-Publication").cast(IntegerType()))
    encoder = OneHotEncoder(inputCols=["bookIdNumber"], outputCols=['bookIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)


def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)


def multiHotEncoderExample(booksSamples):
    samplesWithGenre = booksSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",
                                                                                  F.col("genreIndex").cast(IntegerType()))
    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))
    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    finalSample.printSchema()
    finalSample.show(100)


def ratingFeatures(ratingSamples):
    ratingSamples.printSchema()
    ratingSamples.show()
    # calculate average movie rating score and rating count
    movieFeatures = ratingSamples.groupBy('ISBN').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("Book-Rating").alias("avgRating"),
                                                         F.variance('Book-Rating').alias('Book-Rating-Var')) \
        .withColumn('ratingCountVec', udf(lambda x: Vectors.dense(x), VectorUDT())('ratingCount'))
    movieFeatures.show(10)
    # bucketing
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="avgRating", outputCol="avgRatingBucket")
    # Normalization
    ratingScaler = MinMaxScaler(inputCol="ratingCountVec", outputCol="scaleRatingCount")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(100)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = '../../resources'
    booksResourcesPath = file_path + "/BX-Books.csv"
    booksSamples = spark.read.format('csv').option('header', 'true').load(booksResourcesPath)
    print("Raw books Samples:")
    booksSamples.show(10)
    booksSamples.printSchema()
    print("OneHotEncoder Example:")
    oneHotEncoderExample(booksSamples)
    # print("MultiHotEncoder Example:")
    # multiHotEncoderExample(booksSamples)
    print("Numerical features Example:")
    ratingsResourcesPath = file_path + "/BX-Book-Ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingFeatures(ratingSamples)