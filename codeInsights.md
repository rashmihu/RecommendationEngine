## Code Insights

The code is run on the Snowmane server.

### 1 yelp_dataset/
All the raw datasets used for this project are stored here in the json format. It contains 3 datasets namely user.json , business.json and review.json.

### 2 yelp_etl/
All the exported parquet files are stored in this folder.

### 3 user_etl.py
The data from user.json is read and divided into 80 partitions. This is done to ensure efficient reading of data and for parallel execution of these partitions during computation intensive tasks.
Next, the String Indexer module from Pyspark is used to convert the 22-character long “user_id” into a unique numeric “user_id_index” in order to ease the process of uniquely identifying and searching the users. The next step is to convert the friend’s data from wide format to long format and map the list of users to their corresponding friends(friends_id_index) by making use of the StringIndexerModel. After mapping we convert the list of users back to long format.

<br/>The next step is to aggregate the different complement columns into a single complement value called total_compliments column. We calculate a numeric value for “yelping_since” by subtracting the current date from the specified date. Then we select user_id, user_id_index, review_count, name, friends, friends_id_index, size(friends), average stars, yelping_since and total_compliments and join this with the friend_id_index. We create a temporary view for user_id_index and friends_id_index to use this as an input edgelist and export it as a .txt file to use as an input to the CODA algorithm. We merge the data from the friends_id_index with the user_id_index to create one comprehensive user table and export it into parquet format and is stored in yelp_etl/user_etl. 
        
### 4 business_etl.py
The data from business.json is read and divided into 80 partitions. The records with NA values are dropped and the data is filtered. The String Indexer module from Pyspark is used to convert the 22-character long business_id into a unique business_id_index in order to ease the process of uniquely identifying and searching the business. The average distances of the business in each city is calculated and all the businesses which are outside the average distance are filtered out. The filtered business data is exported into parquet format and is stored in yelp_etl/business_etl.
        
### 5  review_etl.py
The data from review.json is read and divided into 400 partitions. The user_etl and the business_etl parquet is read. The review data is is merged with the user data based on the user_id and the user_id is replaced by user_id_index and similarly this merged review data is merged with the business data based on the business_id and the business id is replaced with the business_id_index and is renamed as business_id. This merged data is the super data and is exported into parquet format and is stored in the yelp_etl/review_etl.
        
### 6 review_transform.py
The review_etl parquet file is read. The data is cleaned by converting the text to lowercase and removing tags, punctuations, whitespaces, numerics and stop words. The cleaned data is tokenized and is modeled using word2vec to convert the review data into feature vectors. We are using MLlib tokenizer and generate review vectors with different dimensions or the K values. We have generated review vectors with dimensions equal to 20, 40, 60, 80 and 100. These different set of k values are used for the purpose of experimentation and comparison with different models. These review vectors are exported into parquet format and is stored in yelp_etl/review_w2v.
        
### 7 Community Detection
The SNAP module is installed in the Downloads folder (/home/r_hassan/Downloads/snap). The CODA algorithm code is written in C++ and is present in the /home/r_hassan/Downloads/snap/examples/coda folder. Do “make all” to create the binary executables. 

<br/>The coda binary is invoked with the following parameters:
./coda -i:”/inputUndirectedGraph.txt” -g:1 -nt:11<br/>
-i: specifies the path to the input file<br/>
-g:specifies the whether the graph is undirected(1) or directed(0) <br/>
-nt:specifies the number of threads to execute in parallel <br/>

If we don't specify the number of communities to detect the code will automatically default to find max of 100 communities. The output is stored in cmtyvv.out.txt located at /home/r_hassan/Downloads/snap/examples/coda. It contains the 100 communities of the users.
        
### 8 Review_rename.py
We are renaming the generated word2vec review vector’s stars column as label and word2vec column as features. We have created a pipeline using spark MLlib using Training and validation split, estimator as the linear regression algorithm and evaluator. The evaluator by default takes the column named label and features as an input. So, we are renaming the output of word2vec results similar to the evaluator. 
        
### 9 User_friends_generator.py                                                                                                                                                                                                                                                                                                                                                                                                            
We need to create a user and friends list to form communities of users using CODA. So, we are creating two sets of user friends lists based on the review count. First, we filter the reviews which have review count>20 and generate a list of users and friends to find the communities of users who have written reviews more than 20. This is done to evaluate how the model behaves with the difference in the number of communities. Similarly, we get the list of users and friends for the users who have written more than 100 reviews and generate different numbers of communities using CODA. We are performing experiments to calculate the relation between the number of communities and the prediction accuracy. 
        
### 10 Review_linear_regr_community.py
Firstly, we load L2 normalized blended community’s data and then we filter the data to obtain all the users who belong to that particular community. We split the review vectors of the filtered set of users and their corresponding rating into training and test datasets and use the training dataset for training the LR model.  All the models to which each user belongs in the testing datasets are combined together to evaluate the performance of our linear regression model. 
        
### 11 Review_rf_community.py
We load the L2 normalized review features that are updated with the community information and randomly split the dataset in 80% training dataset and 20% testing dataset and use the random forest regressor as our estimator in the Training Validation Split model with the number of trees as 20 and maximum depth as 10. We train the model with the training dataset and we fit the model to the training data. We use this trained random forest regressor model to predict the ratings of the reviews by giving the blended features of the users and the businesses and the community to which the user belongs. The predicted ratings from all the models to which the user belongs are combined to predict the final rating. We then calculate the model prediction accuracy by comparing the predicted ratings with the actual ratings.
