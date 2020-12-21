import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import numpy as np
from pyspark.ml.stat import Summarizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Normalizer
spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

normalize = udf(lambda v: v/sum(v), VectorUDT())
mult = udf(lambda x,y: x*y, VectorUDT())

for vector_size in range(20,25,20):
    inputs = "yelp_etl/review_Gt100_w2v_k{}_blended_l2Norm".format(vector_size)
    review = spark.read.parquet(inputs).select("user_id", "business_id", "date", "label", "features")
    
    df_user = review.select("user_id","features")
    df_user = df_user.groupBy("user_id").agg(Summarizer.mean(df_user.features).alias("user_features_mean"))       # Equation (16)
    df_user = df_user.withColumn("user_features",normalize(df_user["user_features_mean"]))    # Equation (22)
    df_user = df_user.drop("features")
    df_user = df_user.drop("user_features_mean")
    df_user.printSchema()

    df_business = review.select("business_id","features")
    df_business = df_business.groupBy("business_id").agg(Summarizer.mean(df_business.features).alias("business_features_mean"))       # Equation (16)
    df_business = df_business.withColumn("business_features",normalize(df_business["business_features_mean"]))    # Equation (22)
    df_business = df_business.drop("features")
    df_business = df_business.drop("business_features_mean")
    df_business.printSchema()

    review = review.join(df_user, on=['user_id'],how='inner')
    review = review.join(df_business, on=['business_id'],how='inner')
    review = review.na.drop()
    review = review.withColumn("blended_features_mult",mult(review["user_features"], review["business_features"]))  # Equation (22)
    review = review.withColumn("blended_features",normalize(review["blended_features_mult"]))  # Equation (22)
    review = review.drop("blended_features_mult")
    review = review.na.drop()
    review.printSchema()
    review.write.parquet("yelp_etl/review_Gt100_w2v_k{}_blended".format(vector_size), mode="overwrite")

