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

for vector_size in range(40,105,20):
    inputs = "yelp_etl/review_Gt100_w2v_k{}_blended".format(vector_size)
    review = spark.read.parquet(inputs)
    review = review.drop("user_features")
    review = review.drop("business_features")
    review = review.drop("blended_features")

    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
    review = normalizer.transform(review)


    df_user = review.select("user_id","normFeatures")
    df_user = df_user.groupBy("user_id").agg(Summarizer.mean(df_user.normFeatures), Summarizer.count(df_user.normFeatures))       # Equation (16)
    df_user = df_user.withColumn("sum(normFeatures)",mult(df_user["mean(normFeatures)"], df_user["count(normFeatures)"]))
    normalizer = Normalizer(inputCol="sum(normFeatures)", outputCol="user_features", p=2.0)
    df_user = normalizer.transform(df_user)                                        # Equation (17)
    df_user = df_user.drop("sum(normFeatures)")
    df_user = df_user.drop("mean(normFeatures)")
    df_user = df_user.drop("count(normFeatures)")
    df_user.printSchema()

    df_business = review.select("business_id","normFeatures")
    df_business = df_business.groupBy("business_id").agg(Summarizer.mean(df_business.normFeatures), Summarizer.count(df_business.normFeatures))       # Equation (16)
    df_business = df_business.withColumn("sum(normFeatures)",mult(df_business["mean(normFeatures)"], df_business["count(normFeatures)"]))
    normalizer = Normalizer(inputCol="sum(normFeatures)", outputCol="business_features", p=2.0)
    df_business = normalizer.transform(df_business)                                        # Equation (17)
    df_business = df_business.drop("sum(normFeatures)")
    df_business = df_business.drop("mean(normFeatures)")
    df_business = df_business.drop("count(normFeatures)")
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
    review.write.parquet(inputs+"_l2Norm", mode="overwrite")

