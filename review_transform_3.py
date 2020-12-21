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

#normalize = udf(lambda v: v/sum(v), VectorUDT())
mult = udf(lambda x,y: x*y, VectorUDT())

for vector_size in range(40,65,20):
    inputs = "yelp_etl/review_Gt100_w2v_k{}_blended_l2Norm".format(vector_size)
    review = spark.read.parquet(inputs).select("user_id", "business_id", "date", "label", "features")
    
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
    review = normalizer.transform(review)
    
    df_user = review.select("user_id","features")
    df_user = df_user.groupBy("user_id").agg(Summarizer.mean(df_user.features).alias("user_features"))       # Equation (16)
    df_user.drop("features")
    df_user.printSchema()

    df_business = review.select("business_id","features")
    df_business = df_business.groupBy("business_id").agg(Summarizer.mean(df_business.features).alias("business_features"))       # Equation (16)
    df_business.drop("features")
    df_business.printSchema()

    review = review.join(df_user, on=['user_id'],how='inner')
    review = review.join(df_business, on=['business_id'],how='inner')
    review = review.na.drop()
    review = review.withColumn("blended_features",mult(review["user_features"], review["business_features"]))  # Equation (22)
    normalizer = Normalizer(inputCol="blended_features", outputCol="blended_features_norm", p=2.0)
    review = normalizer.transform(review)
    review = review.drop("blended_features")
    review = review.withColumnRenamed("blended_features_norm","blended_features")                       # Equation (23)
    review.printSchema()
    review.write.parquet(inputs+"_3", mode="overwrite")

