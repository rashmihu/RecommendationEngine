# An Online Recommendation System Based on Social Community Dynamics
## Introduction 

A hybrid recommendation model is studied [[1]](https://ieeexplore.ieee.org/document/8635542) and is implemented in this work. Improvements are suggested to utilize diverse personalized information to improve the recommendation accuracy. This model makes recommendations based on rating, review texts and social communities. Firstly, we try to implement the hybrid model proposed using PySpark with the larger yelp dataset to study the effects of increase in the size of training and testing data to the measure the accuracy of the model. Secondly, we observe the effects of changing the word2vec dimensions (K), number of communities detected using CoDA (C) and considering the training and testing data which is split by date rather than randomly splitting the data into 80% training and 20% testing to the accuracy of the model. Thirdly build our model using random forest which is a nonparametric machine learning model and compare the prediction accuracy with the linear regression model.

## About Me
 I am Rashmi Hassan Udaya Kumar. I am a graduate student in the department of computer science from the University of North Carolina At Greensboro. This project was developed my research work under the guidence of [Dr. Jing Deng](https://www.uncg.edu/cmp/faculty/j_deng/).

## Technologies and Packages used
Python, Shell, Pyspark 2.4.5, Hadoop 2.4.7, pandas, numpy, scipy sparksql and MLLib

## Implementation Details
![](/images/implementationDetails.png)
## Project Structure
![](/images/projectStructure.png)
## Coclusion
To improve the recommendation accuracy of the hybrid model we need to consider the vector dimensionality of the review vectors as well as the number of communities in the social network.With the large yelp dataset, we observed L2 normalization of review feature vectors and blended vectors achieves better prediction accuracy. From the experiments conducted, we observe that the best recommendation accuracy based on RMSE and MAE is achieved with Word2Vec feature dimension of 40, number of communities as 15 and trained with Random Forest regression. We also observed that splitting the dataset into training and testing ordered by date works better when
using Word2Vec with linear regression model.
