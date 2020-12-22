import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import gensim.parsing.preprocessing as gsp
from gensim import utils
import numpy as np

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

inputs = "yelp_etl/review_w2v"
review = spark.read.parquet(inputs)

review = review.select(col("stars").alias("label"), col("w2v_result").alias("features"))
review.write.parquet("yelp_etl/w2v_renamed.parquet", mode="overwrite")
