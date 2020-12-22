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

for vector_size in range(40,45,20):
    inputs = "yelp_etl/review_Gt100_w2v_k{}_blended_l2Norm".format(vector_size)
    model_save_path = "yelp_etl/RFModel_Gt100_k{}_blended_l2Norm".format(vector_size)
    summary_save_path = "summary_file_k{}_blended_l2Norm_rf.txt".format(vector_size)
    #result_save_path = "yelp_etl/review_Gt100_lROutput_k{}_blended_l2Norm_date_result".format(vector_size)
    result_describe_save_path = "result_stat_Gt100_k{}_blended_l2Norm_rf.csv".format(vector_size)
    review = spark.read.parquet(inputs)
    review = review.select("user_id", "business_id", "review_id", "date", "label", "normFeatures","blended_features")
    # randomly split the data 80% 20%
    train, test = review.randomSplit([0.8, 0.2], seed=12345)

    #split the data based on date by 80% 20%
    # data_win = Window.partitionBy().orderBy('date')
    # review = review.withColumn('percent_rank', percent_rank().over(data_win))
    # train = review.filter(col('percent_rank') <= 0.8)
    # test = review.filter(col('percent_rank') > 0.8)

    train = train.withColumnRenamed("normFeatures","features")
    test = test.withColumnRenamed("blended_features","features")
    rf = RandomForestRegressor() 
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [20]).addGrid(rf.maxDepth, [10]).build()
    #paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01]).addGrid(lr.fitIntercept, [True]).addGrid(lr.elasticNetParam, [0.0]).build()
    
    # 80% of the data will be used for training, 20% for validation.
    tvs = TrainValidationSplit(estimator=rf,estimatorParamMaps=paramGrid,evaluator=RegressionEvaluator(), trainRatio=0.8)
    model = tvs.fit(train)
    # model = lr.fit(train)
    model.write().overwrite().save(model_save_path)
    
    # Print the coefficients and intercept for linear regression
    fh = open(summary_save_path,"a")
    fh.write(" ########## Best Model Params for k= {} ################\n".format(vector_size))
    fh.write("Input File Name:{}\n".format(inputs))
    fh.write(model.bestModel.explainParams())
    fh.write("featureImportances: %s\n" % str(model.bestModel.featureImportances))
    fh.write("treeWeights: %s\n" % str(model.bestModel.treeWeights))
    fh.close()
    
    result = model.transform(test).select("user_id", "business_id", "label", "prediction")
    result = result.withColumn('num_of_reviews',lit(1))
    result = result.groupBy("user_id", "business_id").agg(sum("label").alias("label"), sum("prediction").alias("prediction"), sum("num_of_reviews").alias("num_of_reviews"))
    result = result.withColumn("avg(label)",col("label")/col("num_of_reviews"))
    result = result.withColumn("avg(prediction)",col("prediction")/col("num_of_reviews"))
    result = result.withColumnRenamed("avg(label)","avg_label")
    result = result.withColumnRenamed("avg(prediction)","avg_prediction")
    result.show(1)
       
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="avg_label", predictionCol="avg_prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(result)
    r2 = evaluator.setMetricName("r2").evaluate(result)
    mae = evaluator.setMetricName("mae").evaluate(result)

    fh = open(summary_save_path,"a")
    fh.write("###### avg_label vs avg_prediction ####\n")
    fh.write("Root Mean Squared Error (RMSE) on test data = %g\n" % rmse)
    fh.write("R2 on test data = %g\n" % r2)
    fh.write("MAE on test data = %g\n" % mae)
    fh.close()

    # result.write.parquet("yelp_etl/review_Gt100_RfOutput_k{}_blended_l2Norm_result".format(vector_size), mode="overwrite")
    result.describe().coalesce(1).write.csv(result_describe_save_path,mode="overwrite",header="true")
