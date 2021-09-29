# Databricks notebook source
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode
from math import log, sqrt

# COMMAND ----------

summaries = spark.read.option("header", "false").option("delimiter","\t").csv("/FileStore/tables/plot_summaries.txt")

# COMMAND ----------

summariesDF =summaries.toDF("movieId", "summary")

# COMMAND ----------

tokenizer = RegexTokenizer().setInputCol("summary").setOutputCol("words").setPattern("\\W") 
tokenizedWords = tokenizer.transform(summariesDF)

# COMMAND ----------

stopwords = StopWordsRemover().setInputCol("words").setOutputCol("filtered")
cleanSummary = stopwords.transform(tokenizedWords)

# COMMAND ----------

summaryDF = cleanSummary.select("MovieID","filtered")

# COMMAND ----------

summaryDF.show(5)

# COMMAND ----------

summaryRDD = summaryDF.select(summaryDF.MovieID,explode(summaryDF.filtered)).rdd
summaryRDD.collect()

# COMMAND ----------

wordCount = summaryRDD.map(lambda x: ((x[0], x[1]), 1))

# COMMAND ----------

tfRDD = wordCount.reduceByKey(lambda x, y : x+y)

# COMMAND ----------

tfRDD.take(5)

# COMMAND ----------

tf = tfRDD.map(lambda x : (x[0][1], (x[0][0], x[1])))

# COMMAND ----------

# tf.take(5)

# COMMAND ----------

df = tfRDD.map(lambda x: (x[0][1], 1)).reduceByKey(lambda x,y: x+y)

# COMMAND ----------

df.take(5)

# COMMAND ----------

documentCount = summaryRDD.count()

# COMMAND ----------

idf = df.map(lambda x: (x[0], (x[1], log(documentCount/x[1]))))

# COMMAND ----------

idf.collect()

# COMMAND ----------

tfidf = tf.join(idf).map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1],(x[1][0][1]*x[1][1][1]))))  #token ,(id, tf, df, idf, tfidf)  

# COMMAND ----------

tfidf.collect()

# COMMAND ----------

movieMetadata = spark.read.csv('/FileStore/tables/movie_metadata-ab497.tsv', header=None, sep = '\t')
movieMetadata = movieMetadata.selectExpr("_c0 as movie_id", "_c2 as movie_name")
movieMetadata.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 4.(a) Single Term

# COMMAND ----------

singleTerm = sc.textFile("/FileStore/tables/term.txt").map(lambda x : x.lower())

# COMMAND ----------

term = singleTerm.collect()
term

# COMMAND ----------

keyword = 'action'
# keyword = keywords[0]
highTfMovieIds = tfidf.filter(lambda x: x[0] == keyword).sortBy(lambda x : -x[1][4]).map(lambda x : (x[1][0], x[1][4])).take(10)   
#select top 10 movies sorted by tf-idf score
highTfMovieIds = sc.parallelize(highTfMovieIds)                  # convert to rdd
highTfMovieIds

# COMMAND ----------

movieMetadata = movieMetadata[['movie_id', 'movie_name']].rdd

# COMMAND ----------

result = movieMetadata.join(highTfMovieIds)

result = result.sortBy(lambda x : -x[1][1]).map(lambda x : [x[0],x[1][0],x[1][1]]).toDF(["movie_id", "movie_name", "TF-IDF score"])

# display(result)

# COMMAND ----------

for keyword in term:
  print("Movies which are related to " + keyword + " are :")
  
  #select top 10 movies sorted by tf-idf score
  highTfMovieIds = tfidf.filter(lambda x: x[0] == keyword).sortBy(lambda x : -x[1][4]).map(lambda x : (x[1][0], x[1][4])).take(10)   
  # convert to rdd
  highTfMovieIds = sc.parallelize(highTfMovieIds)                 
  result = movieMetadata.join(highTfMovieIds)
  result = result.sortBy(lambda x : -x[1][1]).map(lambda x : [x[0],x[1][0],x[1][1]]).toDF(["movie_id", "movie_name", "TF-IDF score"])
  result.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 4.(b) Multiple terms

# COMMAND ----------

tfidf.take(10)

# COMMAND ----------

multipleTerms = sc.textFile("/FileStore/tables/query.txt").collect()[0].split()
multipleTerms

# COMMAND ----------

termTf = sc.parallelize(multipleTerms).map(lambda x : (x, 1)).reduceByKey(lambda x,y : x+y)

termTf.collect()

# COMMAND ----------

tfidfmodified = tfidf.map(lambda x :  (x[0], (x[1][1], x[1][4])))

tfidfmodified.take(5)

# COMMAND ----------

joinData = termTf.join(tfidfmodified)
joinData.take(5)

# COMMAND ----------

joinDataModified = joinData.map(lambda x : (x[0], x[1][1][1]))
joinDataModified.take(5)

# COMMAND ----------

tfData = tfidf.map(lambda x : (x[0], (x[1][0], x[1][4]))).join(joinDataModified).map(lambda x : (x[1][0], x[1][1], x[1][0][1]))
tfData.take(5)

# COMMAND ----------

cosineNumerator = tfData.map(lambda x : (x[0], (x[1] * x[2], x[2] * x[2], x[1] * x[1]))).reduceByKey(lambda x,y : ((x[0] + y[0], x[1] + y[1], x[2] + y[2])))
cosineNumerator.take(5)

# COMMAND ----------

cosineScore = cosineNumerator.map(lambda x : (x[0], x[1][0]/(sqrt(x[1][1]) * sqrt(x[1][2]))))
cosineScore.collect()

# COMMAND ----------

results = cosineScore.sortBy(lambda x : -x[1]).map(lambda x : x[0][0])
results.collect()

# COMMAND ----------

resultsRDD =  results.map(lambda x : (x, 1)).reduceByKey(lambda x,y : x+y)
resultsRDD.take(10)

# COMMAND ----------

movieNames = resultsRDD.join(movieMetadata).map(lambda x : (x[0], x[1][1])).toDF(["movie_id", "movie_name"])

movieNames.show(truncate=False)

# COMMAND ----------

movieNames = resultsRDD.join(movieMetadata).map(lambda x : (x[0], x[1][1]))
movieNameScore = cosineScore.map(lambda x : (x[0][0], x[1]))
movieNameScore.join(movieNames).distinct().sortBy(lambda x : -x[1][0]).collect()

# COMMAND ----------

resultsRDD.join(movieMetadata).map(lambda x : x[1]).collect()

# COMMAND ----------

movieMetadata.collect()

# COMMAND ----------

results.filter(lambda x : x[0] == '25532589').collect()
