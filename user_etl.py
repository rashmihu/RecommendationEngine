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
from pyspark.sql.functions import lit, udf, explode
from pyspark.ml.feature import *
from pyspark.sql.functions import collect_set


def drop_from_array_(friend_id, col):
    return [x for x in arr if x != item]

drop_from_array = udf(drop_from_array_, types.ArrayType(types.StringType()))

def main(input):
   user = spark.read.json(input).repartition(80)
   user.show(5)
   indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
   indexer_fitted = indexer.fit(user)
   user_labels = indexer_fitted.labels
   user_indexed = indexer_fitted.transform(user)
   user_indexed.createOrReplaceTempView("user_indexed")
   user_indexed.show(5)

   friends_explode = user_indexed.select("user_id_index",explode("friends").alias("friends"))
   friends_indexer = StringIndexerModel.from_labels(user_labels, inputCol="friends", outputCol="friends_index", handleInvalid="skip")
   friends_indexed = friends_indexer.transform(friends_explode)
   friends_indexed_selected = friends_indexed.select('user_id_index','friends_index')
   friends_indexed_selected.coalesce(1).write.csv('yelp_etl/friends_indexed_selected',sep = ' ', mode="overwrite")
   friends_indexed.createOrReplaceTempView("friends_indexed")
   friends_indexed.show(5)
   friends_indexed_grouped = friends_indexed.groupBy('user_id_index').agg(collect_set('friends_index').alias('friends_index'))
   friends_indexed_grouped.createOrReplaceTempView('friends_indexed_grouped') 
   friends_indexed_grouped.show(5)
   u_etl = spark.sql("SELECT u.user_id, u.user_id_index, u.review_count, u.name, u.friends,f.friends_index, size(u.friends) as num_of_friends, DATEDIFF(current_date(),u.yelping_since) as yelping_since, u.fans, size(u.elite) as elite, (u.compliment_writer + u.compliment_profile + u.compliment_plain + u.compliment_photos + u.compliment_note + u.compliment_more + u.compliment_list + u.compliment_hot + u.compliment_funny + u.compliment_funny + u.compliment_cute + u.compliment_cool) AS total_compliments, u.average_stars FROM user_indexed u inner join friends_indexed_grouped f on u.user_id_index = f.user_id_index ")
   u_etl.createOrReplaceTempView("u_etl")
   u_etl.show(5)
   tot_comp = spark.sql("SELECT * from u_etl ORDER BY total_compliments DESC, review_count DESC")
   tot_comp.createOrReplaceTempView("user")
   tot_comp.write.parquet("yelp_etl/user_etl", mode="overwrite")
    

if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp_dataset/"
    user_filepath = data_path + 'user.json'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('recommendation engine').getOrCreate()
    assert spark.version >= '2.3'
    main(user_filepath)
