from pyspark import SparkContext as sc
import pandas as pd
import os
import numpy as np
from pyspark.sql import SQLContext
import json
import pyspark
import sys
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.types import *
from pyspark.sql.functions import lit, udf

def main(inputs):
    review = spark.read.json(inputs).repartition(400)
    user = spark.read.parquet("yelp_etl/user_etl").select("user_id", "user_id_index")
    business = spark.read.parquet("yelp_etl/business_etl").select("business_id","business_id_index")
    review = review.join(user, on=["user_id"], how="inner").drop("user_id").withColumnRenamed("user_id_index","user_id")
    review = review.join(business, on=["business_id"], how="inner").drop("business_id").withColumnRenamed("business_id_index","business_id")
    user = spark.read.parquet("yelp_etl/user_etl").select("user_id_index", "review_count")
    business = spark.read.parquet("yelp_etl/business_etl").select("business_id_index", "review_count")
    review.createOrReplaceTempView("review")
    user.createOrReplaceTempView("user")
    business.createOrReplaceTempView("business")
    reviewGt100 = spark.sql("SELECT r.review_id, r.user_id, r.business_id, r.date, r.stars AS label, r.text, (r.useful+r.funny+r.cool) AS votes FROM review r INNER JOIN user u ON u.user_id_index = r.user_id WHERE u.review_count > 100")
    #reviewGt100.write.parquet("yelp_etl/review_Gt100",mode = "overwrite")
    reviewGt100.createOrReplaceTempView("reviewGt100")
    reviewGt100 = spark.sql("SELECT r.review_id, r.user_id, r.business_id, r.date, r.label, r.text, r.votes FROM reviewGt100 r INNER JOIN business b ON b.business_id_index = r.business_id WHERE b.review_count > 100")	
    reviewGt100.write.parquet("yelp_etl/review_uRCGt100_bRCGt100",mode = "overwrite")



if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp_dataset/"
    user_filepath = data_path + 'review.json'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('recommendation engine').getOrCreate()
    assert spark.version >= '2.3'
    main(user_filepath)
