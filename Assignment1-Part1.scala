// Databricks notebook source
spark.version

// COMMAND ----------

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

// COMMAND ----------

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

// COMMAND ----------

val pipeline = PretrainedPipeline("recognize_entities_dl", lang="en")

// COMMAND ----------

val book = sc.textFile("/FileStore/tables/littleWomen.txt")

// COMMAND ----------

book.collect()

// COMMAND ----------

val dfWithoutSchema = book.toDF()

// COMMAND ----------

dfWithoutSchema.take(10)

// COMMAND ----------

val rows = dfWithoutSchema.collect().map(_.getString(0)).mkString(" ")

// COMMAND ----------

val result = pipeline.annotate(rows)

// COMMAND ----------

val named_entities = result("entities")

// COMMAND ----------

val wordPairs = named_entities.map(word => (word, 1))

// COMMAND ----------

val converted = wordPairs.toArray

// COMMAND ----------

val final_input = sc.parallelize(converted)

// COMMAND ----------

val wordCounts = final_input.reduceByKey((count1, count2) => count1 + count2)

// COMMAND ----------

val sortedCounts = wordCounts.map(word => word).sortBy(_._2, false)

// COMMAND ----------

sortedCounts.take(10)
