import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import gensim.parsing.preprocessing as gsp
from gensim import utils
import numpy as np

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

inputs = "yelp_etl/review_Gt100"
review = spark.read.parquet(inputs)

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
          ]

def clean_text(text):
    s = text
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

text_cleaner = udf(lambda x: clean_text(x), StringType())
review = review.withColumn("clean_text",text_cleaner('text'))

tokenizer = Tokenizer(inputCol="clean_text",outputCol="words")
for vector_size in range(20,25,20):
    word2Vec = Word2Vec(vectorSize=vector_size, minCount=10, inputCol="words", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer,word2Vec])
    result = pipeline.fit(review).transform(review)
    result.write.parquet("yelp_etl/review_Gt100_w2v_k{}".format(vector_size), mode="overwrite")
