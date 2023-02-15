from pyspark import SparkConf
from pyspark.sql import SparkSession


def process_data_set(spark, users_file_path, books_file_path, book_ratings_file_path, data_set_path):
    users_data = spark.read.format("csv").option("header", "true").load(users_file_path)
    users_data.printSchema()
    users_data.show(5)
    books_data = spark.read.format("csv").option("header", "true").load(books_file_path)
    books_data.printSchema()
    books_data.show(5)
    book_ratings_data = spark.read.format("csv").option("header", "true").load(book_ratings_file_path)
    book_ratings_data.printSchema()
    book_ratings_data.show(5)

    cond = [users_data.user_id == book_ratings_data.user_id]
    users_and_book_ratings_join = book_ratings_data.join(users_data, on=cond, how="inner")

    cond = [books_data.ISBN == users_and_book_ratings_join.ISBN]
    users_and_books_and_book_ratings_join = users_and_book_ratings_join.join(books_data, on=cond, how="inner")

    users_and_books_and_book_ratings_join.printSchema()
    users_and_books_and_book_ratings_join.show(5)
    # 保存方式为覆盖
    # 保存csv文件时去除success文件
    # 指定分隔符为','
    # 保存表列名
    users_and_books_and_book_ratings_join.select("book_ratings_id", users_data["user_id"],
    "location", "age", "book_id", users_and_book_ratings_join["ISBN"], "book_title", "book_author",
    "year_of_publication", "publisher", "book_rating", "timestamp")\
        .write.mode("append")\
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")\
        .option("delimiter", ",")\
        .options(header="true")\
        .csv(data_set_path, sep=',')


if __name__ == '__main__':
    conf = SparkConf().setAppName('dataSet').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    FILEPATH = "../../resources/csv/"
    RAWDATAPATH = "rawdata/"
    DATASETPATH = "dataset/"
    USERSFILEPATH = FILEPATH + RAWDATAPATH + "users.csv"
    BOOKSFILEPATH = FILEPATH + RAWDATAPATH + "books.csv"
    BOOKRATINGSFILEPATH = FILEPATH + RAWDATAPATH + "book_ratings.csv"
    process_data_set(spark, USERSFILEPATH, BOOKSFILEPATH, BOOKRATINGSFILEPATH, FILEPATH+DATASETPATH)
