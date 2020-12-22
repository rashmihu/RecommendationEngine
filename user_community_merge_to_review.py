import sys
from pyspark import SparkConf, SparkContext
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, row_number, lit, explode, collect_set
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField, StringType

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

for comm in range(30,31,5):
    community_data = r"yelp_etl/userFriendsListGt100/Gt100_c{}cmtyvv.out.txt".format(comm)
    myschema = StructType([StructField("user_id_arr", StringType(), False)])
    df_communities = spark.read.csv(community_data, schema=myschema)
    df_communities = df_communities.withColumn("user_id_arr", split(col("user_id_arr"),"\t\s*").cast(ArrayType(DoubleType())).alias("user_id_arr"))
    w = Window.orderBy(lit(1))
    df_communities = df_communities.withColumn("community_id", row_number().over(w))
    df_communities = df_communities.select("community_id",explode("user_id_arr").alias("user_id"))
    number_of_communities = df_communities.agg({"community_id":"max"}).collect()[0][0]
    df_communities = df_communities.groupBy('user_id').agg(collect_set('community_id').alias('community_id_arr'))
    df_communities = df_communities.na.drop()
    df_communities.show(5)
    df_communities.printSchema()
    for vector_size in range(40, 45, 20):
        df_review = spark.read.parquet("yelp_etl/review_Gt100_w2v_k{}_blended_l2Norm".format(vector_size))
        df_review = df_review.drop("features")
        df_review = df_review.withColumnRenamed("normFeatures", "features")
        df_review = df_review.select("user_id", "business_id", "review_id", "date", "label", "features", "blended_features")
        df_review = df_review.join(df_communities,on=["user_id"],how="inner")
        df_review = df_review.na.drop()
        df_review.write.parquet("yelp_etl/review_Gt100_w2v_k{}_blended_communities_c{}_l2Norm".format(vector_size, comm),mode="overwrite")
