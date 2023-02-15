from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id


def process_users_file(spark, users_file_path, raw_data_path):
    bx_users_data = spark.read.format("csv").option("header", "true").load(users_file_path)
    bx_users_data.printSchema()
    bx_users_data.show(5)
    bx_users_data = bx_users_data.withColumnRenamed("User-ID", "user_id")
    bx_users_data = bx_users_data.withColumnRenamed("Location", "location")
    bx_users_data = bx_users_data.withColumnRenamed("Age", "age")
    bx_users_data.show(5)
    # 保存方式为覆盖
    # 保存csv文件时去除success文件
    # 指定分隔符为','
    # 保存表列名
    bx_users_data.select("user_id", "location", "age")\
        .write.mode("append")\
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")\
        .option("delimiter", ",")\
        .options(header="true")\
        .csv(raw_data_path, sep=',')


def process_books_file(spark, books_file_path, raw_data_path):
    bx_books_data = spark.read.format("csv").option("header", "true").load(books_file_path)
    bx_books_data.printSchema()
    bx_books_data.show(5)
    bx_books_data_with_id = bx_books_data.withColumn("book_id", monotonically_increasing_id()+1)
    bx_books_data_with_id.printSchema()
    bx_books_data_with_id.show(5)
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Book-Title", "book_title")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Book-Author", "book_author")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Year-Of-Publication", "year_of_publication")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Publisher", "publisher")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Image-URL-S", "image_url_s")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Image-URL-M", "image_url_m")
    bx_books_data_with_id = bx_books_data_with_id.withColumnRenamed("Image-URL-L", "image_url_l")
    # 保存方式为覆盖
    # 保存csv文件时去除success文件
    # 指定分隔符为','
    # 保存表列名
    bx_books_data_with_id.select("book_id", "ISBN", "book_title", "book_author", "year_of_publication", "publisher",
    "image_url_s", "image_url_m", "image_url_l")\
        .write.mode("append")\
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")\
        .option("delimiter", ",")\
        .options(header="true")\
        .csv(raw_data_path, sep=',')


def process_book_rarings_file(spark, ratings_file_path, book_ratings_file_path, file_path):
    ratings_data = spark.read.format("csv").option("header", "true").load(ratings_file_path)
    ratings_data.printSchema()
    ratings_data.show(5)
    ratings_data_with_id = ratings_data.withColumn("ID", monotonically_increasing_id()+1)
    ratings_data_with_id.printSchema()
    ratings_data_with_id.select("ID", "userId", "movieId", "rating", "timestamp").show(5)

    book_ratings_data = spark.read.format("csv").option("header", "true").load(book_ratings_file_path)
    book_ratings_data.printSchema()
    book_ratings_data.show(5)
    book_ratings_data_with_id = book_ratings_data.withColumn("book_ratings_id", monotonically_increasing_id()+1)
    book_ratings_data_with_id.printSchema()
    book_ratings_data_with_id.select("book_ratings_id", "User-ID", "ISBN", "Book-Rating").show(5)

    cond = [ratings_data_with_id.ID == book_ratings_data_with_id.book_ratings_id]
    book_ratings_join = ratings_data_with_id.join(book_ratings_data_with_id, on=cond, how="inner")
    book_ratings_join.select("book_ratings_id", "User-ID", "ISBN", "Book-Rating", "timestamp").show(10)
    book_ratings_join = book_ratings_join.withColumnRenamed("User-ID", "user_id")
    book_ratings_join = book_ratings_join.withColumnRenamed("Book-Rating", "book_rating")

    # 保存方式为覆盖
    # 保存csv文件时去除success文件
    # 指定分隔符为','
    # 保存表列名
    book_ratings_join.select("book_ratings_id", "user_id", "ISBN", "book_rating", "timestamp")\
        .write.mode("append")\
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")\
        .option("delimiter", ",")\
        .options(header="true")\
        .csv(file_path, sep=',')


if __name__ == '__main__':
    conf = SparkConf().setAppName('dataStructure').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    FILEPATH = "../../resources/csv/"
    RAWDATAPATH = "rawdata/"
    RAWDATAFROMBXPATH = "rawdatafrombx/"
    RAWDATAFROMMOVIELENSPATH = "rawdatafrommovielens/"
    RATINGSFILEPATH = FILEPATH + RAWDATAFROMMOVIELENSPATH + "ratings.csv"
    BOOKRATINGSFILEPATH = FILEPATH + RAWDATAFROMBXPATH + "BX-Book-Ratings.csv"
    BOOKSFILEPATH = FILEPATH + RAWDATAFROMBXPATH + "BX-Books.csv"
    USERSFILEPATH = FILEPATH + RAWDATAFROMBXPATH + "BX-Users.csv"
    BOOKRATINGSFILENAME = "book_ratings.csv"
    process_users_file(spark, USERSFILEPATH, FILEPATH+RAWDATAPATH)
    process_books_file(spark, BOOKSFILEPATH, FILEPATH+RAWDATAPATH)
    process_book_rarings_file(spark, RATINGSFILEPATH, BOOKRATINGSFILEPATH, FILEPATH+RAWDATAPATH)
