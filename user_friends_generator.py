import sys
from pyspark import SparkConf, SparkContext
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

df_user = spark.read.parquet("yelp_etl/user_etl")

df_userGt20 = df_user.filter((df_user.review_count>20))
df_userGt100 = df_userGt20.filter((df_userGt20.review_count>100))

df_userGt20_long = df_userGt20.select("user_id_index",explode("friends_index").alias("friends"))
df_userGt100_long = df_userGt100.select("user_id_index",explode("friends_index").alias("friends"))

df_userGt20_long.coalesce(1).write.csv('yelp_etl/userFriendsListGt20',sep = ' ', mode="overwrite")
df_userGt100_long.coalesce(1).write.csv('yelp_etl/userFriendsListGt100',sep = ' ', mode="overwrite")
