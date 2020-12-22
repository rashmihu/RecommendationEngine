import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, explode, lit, col, percent_rank
from pyspark.sql.window import Window

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

vector_size = int(sys.argv[1])
i = int(sys.argv[2])
num_of_communities = int(sys.argv[3])
inputs = "yelp_etl/review_Gt100_w2v_k{}_blended_communities_c{}_l2Norm".format(vector_size, num_of_communities )
model_save_path = "yelp_etl/RFModel_Gt100_k{}_blended_l2Norm".format(vector_size)
summary_save_path = "summary_file_RF_k{}_blended_communities_c{}.txt".format(vector_size,num_of_communities )
#result_save_path = "yelp_etl/review_Gt100_lROutput_k{}_blended_l2Norm_date_result".format(vector_size)
if i <= num_of_communities:
    print ("Community {} started".format(i))
    df_review = spark.read.parquet(inputs)
    df_review = df_review.select("user_id", "business_id", "date", explode("community_id_arr").alias("community_id"), "review_id", "label","features", "blended_features")
    df_review.na.drop()
    review = df_review.filter(df_review.community_id == i)
    # randomly split the data 80% 20%
    train, test = review.randomSplit([0.8, 0.2], seed=12345)

    # split the data based on date by 80% 20%
    # data_win = Window.partitionBy("user_id").orderBy('date')
    # review = review.withColumn('percent_rank', percent_rank().over(data_win))
    # train = review.filter(col('percent_rank') <= 0.8)
    # test = review.filter(col('percent_rank') > 0.8)

    test = test.drop("features")
    test = test.withColumnRenamed("blended_features","features")
    rf = RandomForestRegressor() 
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [20]).addGrid(rf.maxDepth, [10]).build()
    
    # 80% of the data will be used for training, 20% for validation.
    tvs = TrainValidationSplit(estimator=rf,estimatorParamMaps=paramGrid,evaluator=RegressionEvaluator(), trainRatio=0.8)
    
    model = tvs.fit(train)
    
    # Print the coefficients and intercept for linear regression
    fh = open(summary_save_path,"a")
    fh.write(" ########## Best Model Params for k= {}, community={} ################\n".format(vector_size, i))
    fh.write("Input File Name:{}\n".format(inputs))
    fh.write(model.bestModel.explainParams())
    fh.write("featureImportances: %s\n" % str(model.bestModel.featureImportances))
    fh.write("treeWeights: %s\n" % str(model.bestModel.treeWeights))
    fh.close()
    
    # model = TrainValidationSplitModel.load("yelp_etl/community_models_l2Norm_c{}/lRModel_Gt100_k{}_blended_community{}".format(num_of_communities, vector_size,i))
    result = model.transform(test).select("user_id", "business_id", "label", "prediction")
    result = result.withColumn('num_of_reviews',lit(1))
    result = result.groupBy("user_id", "business_id").agg(sum("label").alias("label"), sum("prediction").alias("prediction"), sum("num_of_reviews").alias("num_of_reviews"))
    model.save("yelp_etl/community_models_l2Norm_k{}_c{}/RFModel_Gt100_k{}_blended_community{}".format(vector_size,num_of_communities,vector_size,i))
    
    if (i==1):
        result_combined = result
        result_combined.write.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result_odd", mode="overwrite")
    else:
        if(i%2 == 0):
            result_combined = spark.read.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result_odd")
        else:
            result_combined = spark.read.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result_even")
        result_combined = result_combined.union(result)
        result_combined = result_combined.na.drop()
        result_combined = result_combined.groupBy("user_id", "business_id").agg(sum("label").alias("label"), sum("prediction").alias("prediction"), sum("num_of_reviews").alias("num_of_reviews"))
        if(i >= num_of_communities):
            result_combined.write.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result", mode="overwrite")
        elif(i%2 == 0):
            result_combined.write.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result_even", mode="overwrite")
        else:
            result_combined.write.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result_odd", mode="overwrite")
    print ("Community {} ended".format(i))
else:
    result_combined = spark.read.parquet("yelp_etl/review_Gt100_lROutput_blended_community_result")
    result_combined = result_combined.withColumn("avg(label)",col("label")/col("num_of_reviews"))
    result_combined = result_combined.withColumn("avg(prediction)",col("prediction")/col("num_of_reviews"))
    result_combined = result_combined.withColumnRenamed("avg(label)","avg_label")
    result_combined = result_combined.withColumnRenamed("avg(prediction)","avg_prediction")
    result_combined.show(1)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="avg_label", predictionCol="avg_prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(result_combined)
    r2 = evaluator.setMetricName("r2").evaluate(result_combined)
    mae = evaluator.setMetricName("mae").evaluate(result_combined)

    fh = open(summary_save_path ,"a")
    fh.write("Root Mean Squared Error (RMSE) on test data = %g\n" % rmse)
    fh.write("R2 on test data = %g\n" % r2)
    fh.write("MAE on test data = %g\n" % mae)
    fh.close()
    # result_combined.write.parquet("yelp_etl/review_Gt100_lROutput_k{}_blended_community_c{}_result".format(vector_size, num_of_communities), mode="overwrite")
