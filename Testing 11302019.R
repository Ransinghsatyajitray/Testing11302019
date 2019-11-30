
#Predictive modeling and machine learning in R with the caret package

#Powerful and simplified modeling with caret

# The R caret package will make your modeling life easier – guaranteed. 
# caret allows you to test out different models with very little change to 
# your code and throws in near-automatic cross validation-bootstrapping and parameter tuning for free.

# For example, below we show two nearly identical lines of code. Yet they run entirely different models. 
# In the first, method = "lm" tells caret to run a traditional linear regression model. In the second line 
# method = "rf" tells caret to run a random forest model using the same data. One tiny syntax change and 
# you run an entirely new type of model.


# library(caret)
# # In R, the annual_pm~. means use annual_pm as the model response
# # and use all other variables as predictors
# lm1 <- train(annual_pm~., data = air, method = "lm")
# rf1 <- train(annual_pm~., data = air, method = "rf")


#Behind the scenes caret takes these lines of code and automatically resamples the models and conducts parameter tuning


# By simply changing the method argument, you can easily cycle between, for example, running a linear model, a gradient boosting 
# machine model and a LASSO model. In total, there are 233 different models available in caret.


# It’s important to note that behind the scenes, caret is not actually performing the statistics/modeling – this job is left to 
# individual R packages. For example, when you run the two lines of code above caret uses the lm() function from the stats 
# package to compute the linear regression model and the randomForest() function from the randomForest package for the random 
# forest model.


# The beauty of having caret provide the vehicle to run these models is that you can use exactly the same function, train(), to run 
# all of the models. The train() function accepts several caret-specific arguments and you can also provide arguments that get
# fed to the underlying modeling package/function.

# lm1 <- train(annual_pm~., data = air, method = "lm")
# class(lm1)

## [1] "train"         "train.formula"

#attributes(lm1)
## $names
##  [1] "method"       "modelInfo"    "modelType"    "results"     
##  [5] "pred"         "bestTune"     "call"         "dots"        
##  [9] "metric"       "control"      "finalModel"   "preProcess"  
## [13] "trainingData" "resample"     "resampledCM"  "perfNames"   
## [17] "maximize"     "yLimits"      "times"        "levels"      
## [21] "terms"        "coefnames"    "xlevels"     
## 
## $class
## [1] "train"         "train.formula"

#Among these attributes is the finalModel which is the final model from the underlying package used by caret.

#lm1$finalModel
## 
## Call:
## lm(formula = .outcome ~ ., data = dat)
## 
## Coefficients:
##    (Intercept)        latitude       longitude         Kvol100  
## -255.691658494     1.702674197    -2.612063511     0.001096599  
##        pop1000        oil6_500        truck300         road100  
##    0.000000143     0.008935559     0.458994434     0.084027878  
##       p_ind300      p_bldg1000        imperv50  
##   32.966423769     0.291760773     0.016474590

#rf1$finalModel
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: regression
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##           Mean of squared residuals: 2.042014
##                     % Var explained: 58.5

#Easy data splitting
# A common approach in machine learning, assuming you have enough data, is to split your data into a training dataset 
# for model building and testing dataset for final model testing. The model building, with the help of resampling, 
# would be conducted only on the training dataset. Once a final model has been chosen you would do an additional test of 
# performance by predicting the response in the testing set.


# For simple random splitting (or stratified random splitting) you can make use of the createDataPartition() function. You feed the 
# function your outcome, a proportion and specify whether you want the output as a list:

# samp <- createDataPartition(air$annual_pm, p = 0.8, list = FALSE)
# training <- air[samp,]
# testing <- air[-samp,]


# In this example, our outcome is annual air pollution levels (pm refers to particulate matter) which is a continuous outcome 
# in which case the function will randomly split the data based on quartiles of the data. If the first argument to 
# createDataPartition() is categorical caret will perform stratified random sampling on the variable levels.


# The createDataPartition() function returns the index number of the observations to be included in the training dataset and this 
# index number can be used to create your training and testing datasets as we’ve done above.


# Realistic model estimates through built-in resampling
# The ultimate goal of a predictive model is to predict data you’ve never seen before. Your modeling dataset should be 
# representative of other datasets you’re likely to see but the goal is not to “predict” the data you have in hand, but 
# to develop a model that will predict new datasets.



# An R-squared from a model based on the full dataset is unrealistic
# One approach to the development of a predictive model might be to simply identify good predictors and an appropriate
# model type based on your full dataset. You might then use R-squared or root mean square error to evaluate model 
# performance and you might assume that the R-squared you see in your full model would be what you would see when you 
# predict a new dataset.
# 
# Unfortunately, an R-squared based on modeling your full dataset is not a realistic measure of how well you’re 
# likely to perform on a new dataset. Your R-squared from predictions on a new dataset is almost certainly going to be 
# lower. This is due to the fact that with the R-squared based on the full model you are essentially re-predicting the 
# same data you used to build the model in the first place.


library(dplyr)
library(stringr)
library(readr)
library(tidyr)
library(lubridate)
library(purrr)
library(readxl)
library(data.table) #fread
library(ggplot2)
library(ggforce) #this is for paginating the wrap
library(sqldf)
library(rebus)
library(scales)
library(broom)
library(corrplot) #for correlation plot


#Machine Learning Libraries

#Interesting Note:
#___________________
#Every model and every argument in machine learning models startwith small letters 
#and when there is situation of space its written in Capital without any space
#eg.fastAdaboost,caTools,trControl,trainControl,tuneGrid, tuneLength, preProcess

library(caret) #8 major arguments, 1. Y~X, data, model=, trControl=trainControl, tuneGrid or tuneLength, preProcess
library(mlbench)  #for sample data for machine learning
library(fastAdaboost) #AdaBoost Classification Trees
library(ipred)      #Stabilized Linear Discriminant Analysis
library(earth)      #Multivariate Adaptive Regression Spline model
library(caTools)    #Boosted Logistic Regression	model
library(C50)        #C50 ensemble model
library(xgboost)    #Xtreme gradiant boosting
library(h2o)        #glmnet model
library(naivebayes) #For Naivebayes models
library(pls)        #Principal Component Analysis	Models
library(randomForest)# For Random Forest Models
library(rrcov)      #Robust Linear Discriminant Analysis models
library(elasticnet) #elastic-net models
library(gbm)     #for generalised boosted regression models
library(ellipse) #for drawing ellipse in feature plots
library(glmnet)  #For lasso and Elastic-Net Regularized Generalized Linear Model




data("ChickWeight")

ChickWeight


lm1 <- train(weight~., data = ChickWeight, method = "lm")
rf1 <- train(weight~., data = ChickWeight, method = "rf")

class(lm1)
attributes(lm1)

lm1$results
lm1$pred
lm1$metric
lm1$finalModel


lm1$method
lm1$modelInfo
lm1$modelType
lm1$results
lm1$pred
lm1$bestTune
lm1$call
lm1$dots
lm1$metric
lm1$control
lm1$finalModel
lm1$preProcess
lm1$trainingData
lm1$resample
lm1$resampledCM
lm1$perfNames
lm1$maximize
lm1$times
lm1$levels
lm1$terms
lm1$coefnames
lm1$xlevels

str(ChickWeight)

#The models automatically the dummification

attributes(rf1)
rf1$finalModel
rf1$metric


rf1$method
rf1$modelInfo
rf1$modelType
rf1$results
rf1$pred
rf1$bestTune
rf1$call
rf1$dots
rf1$resampledCM
rf1$resample
rf1$control
rf1$perfNames
rf1$maximize
rf1$yLimits  
rf1$times
rf1$levels
rf1$terms
rf1$coefnames
rf1$xlevels



#Easy data splitting
#This can be done by the help of createDataPartition function in caret package
#it randomly split the data

samp <- createDataPartition(ChickWeight$weight, p = 0.8, list = FALSE)



# A common approach in machine learning, assuming you have enough data, is to split your 
# data into a training dataset for model building and testing dataset for final model testing. 
# The model building, with the help of resampling, would be conducted only on the training dataset. 
# Once a final model has been chosen you would do an additional test of performance by predicting the 
# response in the testing set.

training<-ChickWeight[samp,]
testing<-ChickWeight[-samp,]

# If the first argument to createDataPartition() is categorical caret will perform stratified 
# random sampling on the variable levels.

#1st argument Numeric => simple random sampling
#1st argument Categorical => stratified random sampling


#The createDataPartition() function returns the index number of the observations to be 
# included in the training dataset and this index number can be used to create your training 
# and testing datasets as we’ve done above.




#___________________________

#create data partition is better then insample prediction 
#Resampling is better than create data partition

#resampling with repeat cv >resampling with cv > partition with test train split > insample prediction


# To get a more realistic sense of how well you will predict a new dataset you can “pretend” some 
# of your data is a new dataset and set that data aside. Then you can develop a model with the rest of 
# the data then predict the unmodeled data and see how you did. This would be similar to splitting into 
# training and testing datasets.

# Doing this once, though, will only test one possible data split. What if the data you set aside has all 
# of the outliers or none of the outliers? Instead, 
# you can do this over and over again. This process, known as resampling, 
# will give a more realistic sense (general sense) of how well your model will perform on new data.


summary(lm1$finalModel)$r.squared  #This unrealistic R squared (as it doesn't consider the resampling)
#as this is an R squared from the model using all the data when what you 
#want instead is an R squared based on applying your model to the new 
#data which is what you get with resampling

lm1  # We can see th realistic r squared by this



#The caret package does automatic resampling — but what is it doing exactly?

# The output from lm1 above tells you that to compute the realistic R-squared and RMSE caret used bootstrap 
# resampling with 25 repetitions – this is the default resampling approach in caret.


# Randomly sample the data with replacement. This means that a single observation could be chosen more than once. The total size of the modeling dataset will be the same as the original dataset but some sites will not be included and some will be included more than once.
# Develop a model based on the randomly sampled sites only.
# Predict the withheld sites (also known as “out-of-bag” (OOB) observations) and compute an R-squared based on the predictions.
# Repeat the process (25 models are run) and average the R-squared values.
# When the resampling is done, caret then runs a final model on the full dataset and stores this in finalModel. So, in this example, 25 + 1 models are run, 25 bootstraps and one model run using full dataset.


rf1$results

#Bootstrap is the default resampling approach but you can easily use cross validation instead

tc<-trainControl(method="cv",number=10)

#caret can take care of the dirty work of setting up the resampling with the help of the trainControl() function and 
#the trControl argument. Careful here, the argument is trControl not trainControl.

lm1_cv<-train(weight~., data = ChickWeight, method = "lm",trControl=tc)

lm1_cv

#“parameter” refers to a mathematical term or other aspect of the model that has an influence on the model results but the “best” value for this
# parameter cannot be derived analytically 

lm1$results

rf1$results

#For each different value of mtry, the tuning parameter, caret performs a separate bootstrap (or cross validation 
#if this is what you set up). Since three values were tested and the default is a 25 rep bootstrap, this means that 
#caret initially ran the model 75 times.

#Once caret determines the best value of the tuning parameter, in this case the value of 2 is best for mtry 
#(highest R-squared) it runs the model one more time using this value for the tuning parameter and all the observations 
#and this is stored in finalModel.

# For mtry above caret made an educated guess as to which values to test. There will be situations where you want to test more or
# different values. If this is the case you have two options:


#You can use the tuneLength argument in train() which tells caret how many possible values to test. 
#You can use the tuneGrid argument in train() to tell caret exactly which values to test.

#The tuneLength parameter tells the algorithm to try different default values for the main parameter
#eg, by default rf1 considered 3 mtry but we can change it to 4 using tuneLength
#tuneLength automatic but inefficient in some cases


#The tuneGrid parameter lets us decide which values the main parameter will take
#if we want to manually control the mtry then we use tuneGid and which values to test
#tuneGrid manual but efficient in some cases 

#tuneGrid
tg<-data.frame(mtry=seq(2,10,by=2))

rf1_1<-train(weight~., data = ChickWeight, method = "rf",tuneGrid=tg)
rf1_1$results


rf1_2<-train(weight~., data = ChickWeight, method = "rf",tuneLength= 7)
rf1_2$results

#The order of all model (default, rf with tuneGrid or rf with tuneLength) is in decreasing order of RMSE




#Easy comparison of models
#The final amazing feature of caret we want to point out is how easy it is to compare different types of models.
#Once you’ve created the different models, you will want to compare their performance.


#In caret you can compare models in a couple of ways.
#1. you can stack the models together in a list and then compare the results qualitatively using the function resamples().
#2. Alternatively, if you want to quantitatively test if one model is better than another you can use compare_models().



# 1. (name_to_list<- list(alias_name_to_model1,alias_name_to_model2,...) + resamples(name_to_list)
# 2. compare_models(model1, model2)


#1.
model_list <- list(lm = lm1, rf = rf1)
res <- resamples(model_list)
summary(res)



# boxplot comparison (bw means box whisker plot)
bwplot(res)
# Dot-plot comparison
dotplot(res)

#In the models rf has the lowest Mean RMSE and the highest Rsquared



#2.
compare_models(lm1, rf1) 
#from the p value (0.1) at 95 % CL it is still, we cannot confirm rf is significatly different from the lm




#Shrinkage/regularization models with caret
#+++++++++++++++++++++++++++++++++++++++++++++

# Here we run a shrinkage/regularization model (method = "glmnet") which has two tuning parameters alpha and lambda. 
# If alpha is set to 0 this process runs a ridge model, if it’s set to 1 it runs a LASSO model and an alpha between 0 
# and 1 results in an elastic net model.'


#Take note of the preProcess argument below, caret makes it super-easy to center/scale your data or apply other transformation like BoxCox.



#How to Evaluate Machine Learning Algorithms with R

# The most robust way to discover good or even best algorithms for your dataset is by trial and error. Evaluate a diverse set of algorithms on your dataset and see what works and drop what doesn’t.
# 
# I call this process spot-checking algorithms.


# load libraries
#library(mlbench)
#mlbench contain datasets for machine learning benchmark problems


#library(caret)

# load data
data(PimaIndiansDiabetes) #coming from ml bench package
# rename dataset to keep code below generic
dataset <- PimaIndiansDiabetes

control <- trainControl(method="repeatedcv", number=10, repeats=3)

seed <- 7

metric <- "Accuracy"



# Test Metric
# There are many possible evaluation metrics to choose from. 
# Caret provides a good selection and you can use your own if needed.
# __________________________________________________________________________
# Some good test metrics to use for different problem types include:
#   
# Classification:
# +++++++++++++++++  
# 1. Accuracy: x correct divided by y total instances. Easy to understand and widely used.
# 2. Kappa: easily understood as accuracy that takes the base distribution of classes into account.

# Regression:
# ++++++++++++++++++  
# 1. RMSE: root mean squared error. Again, easy to understand and widely used.
# 2. Rsquared: the goodness of fit or coefficient of determination.
# 3. Other popular measures include ROC and LogLoss.


# 2. Model Building
# There are three concerns when selecting models to spot check:
#   
# 1. What models to actually choose.
# 2. How to configure their arguments.
# 3. Preprocessing of the data for the algorithm.



# Algorithms
# It is important to have a good mix of algorithm representations (lines, trees, instances, etc.) as well as algorithms for learning those representations.
# 
# A good rule of thumb I use is “a few of each”, for example in the case of binary classification:
#   
# Linear methods: Linear Discriminant Analysis and Logistic Regression.
# Non-Linear methods: Neural Network, SVM, kNN and Naive Bayes
# Trees and Rules: CART, J48 and PART
# Ensembles of Trees: C5.0, Bagged CART, Random Forest and Stochastic Gradient Boosting

#How many algorithms? At least 10-to-20 different algorithms.


#Algorithm Configuration
#+++++++++++++++++++++++
#When we are spot checking, we do not want to be trying many variations of algorithm parameters, 
#that comes later when improving results.


#Data Preprocessing
#++++++++++++++++++
#many instance based algorithms work a lot better if all input variables have the same scale.
#Fortunately, the train() function in caret lets you specify preprocessing of the data to perform prior to training. 
#The transforms you need are provided to the preProcess argument as a list and are executed on the data sequentially


#preProcess=c("center", "scale")


#Algorithm Spot Check


# Linear Discriminant Analysis  (e1071 package installation required)
set.seed(seed)
fit.lda <- train(diabetes~., data=dataset, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(diabetes~., data=dataset, method="glm", metric=metric, trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(diabetes~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(diabetes~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Naive Bayes  ( klaR   Package required) 
set.seed(seed)  
fit.nb <- train(diabetes~., data=dataset, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(diabetes~., data=dataset, method="rpart", metric=metric, trControl=control)
# C5.0
set.seed(seed)
fit.c50 <- train(diabetes~., data=dataset, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(diabetes~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(diabetes~., data=dataset, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling) (gbm package required)
set.seed(seed)
fit.gbm <- train(diabetes~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)


##1.
model_list <- list(Linear_Discriminant_Analysis=fit.lda,
                   Logistic_Regression=fit.glm,
                   SVM_Radial=fit.svmRadial,
                   knn=fit.knn,
                   Naive_Bayes=fit.nb,
                   CART=fit.cart,
                   C50_Model=fit.c50,
                   Bagged_CART=fit.treebag,
                   Random_Forest=fit.rf,
                   stochastic_Gradient_Boosting=fit.gbm)
res <- resamples(model_list)
summary(res)



# boxplot comparison
bwplot(res)
# Dot-plot comparison
dotplot(res)


#From the graph it seemed Logistic regression, followed by Linear_Discriminant_Analysis followed by SGB are better for the data


#Now perform more indepth analysis into the model with feature selection/engineering,finally predict the best model into the validation data and 
#create confusion matrix




#How do you optimize a machine learning model after new data comes in?
# Most models in python or R libraries need to be retrained from scratch. Bayesian models can, in theory, incorporate new observations sequentially, but in practice it is often the case that they are retrained (or the posteriors are approximated by analytic distributions and then used as priors in the next round of fitting. Point is, the model needs to be refit one way or another).
# 
# It might be beneficial to monitor your model's performance over time. 
#Once performance starts to degrade, you can retrain your model with more recent data. 
#Alternatively, you can schedule models to retrain quarterly or yearly.


# So, I have not been able to find any literature on this subject but it seems like something worth giving a thought:
#   
#   What are the best practices in model training and optimization if new observations are available?
#   
#   Is there any way to determine the period/frequency of re-training a model before the predictions begin to degrade?
#   
#   Is it over-fitting if the parameters are re-optimised for the aggregated data?


# Prior to deploying a model to production data scientists go through a rigorous process of model validation which includes:
#   
# Assembling datasets – Gathering datasets from different sources such as different databases.
# Feature Engineering – Deriving columns from raw data that will improve predictive performance.
# Model Selection – Comparing different learning algorithms.
# Error Estimation – Optimizing over a search space to find the best model and estimate its generalization error.



#How often should you retrain your model
#So far we’ve discussed what model drift is and a number of ways to identify it. 
#So the question becomes, how do we remedy it? If a model’s predictive performance has fallen due to changes in the environment, the solution is to retrain the model on a new training set which reflects the current reality.


#How often should you retrain your model? And how do you determine your new training set? 
#As with most difficult questions, the answer is that it depends. But what does it depend on?

# Sometimes the problem setting itself will suggest when to retrain your model. For instance, 
# suppose you’re working for a university admissions department and are tasked with building a 
# student attrition model that predicts whether a student will return the following semester. 
# This model will be used to generate predictions on the current cohort of students directly 
# after midterms. Students identified as being at risk of churning will automatically be enrolled 
# in tutoring or some other such intervention.

#This is an example of a periodic retraining schedule.



# If the threshold is too low you risk retraining too often which can result in high costs associated with the cost of compute. If the threshold is too high you risk 
# not retraining often enough which leads to suboptimal models in production. 

#How can you retrain your model?

# If you decide to retrain your model periodically, then batch retraining is perfectly sufficient. 
# This approach involves scheduling model training processes on a recurring basis using a job scheduler 
# such as Jenkins or Kubernetes CronJobs.

# A machine learning model’s predictive performance is expected to decline as soon as the model is deployed 
# to production. For that reason it’s imperative that practitioners prepare for degraded performance by setting 
# up ML-specific monitoring solutions and workflows to enable model retraining. While the frequency of retraining
# will vary from problem-to-problem, ML engineers can start with a simple strategy that retrains models on a 
# periodic basis as new data arrives and evolve to more complex processes that quantify and react to model drift.



#Feature Selection In R
#+++++++++++++++++++++++++

#Once our model is ready and before doing the modeling we will go for feature selection
# Selecting the right features in your data can mean the difference between mediocre performance with long training times and 
# great performance with short training times.

#The caret R package provides tools to automatically report on the relevance and importance of attributes in your data and even 
#select the most important features for you.


# How to remove redundant features from your dataset.
# How to rank features in your dataset by their importance.
# How to select features from your dataset using the Recursive Feature Elimination method.



#Remove Redundant Features
#+++++++++++++++++++++++++++
#Identify highly correlated features

# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
data(PimaIndiansDiabetes)
# calculate correlation matrix
correlationMatrix <- cor(PimaIndiansDiabetes[,1:8])


library(corrplot)
corrplot(correlationMatrix, method="color")

# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)  #How many are correlated (It seems only the diagonal elements are picked)



#Rank Features By Importance
#++++++++++++++++++++++++++++


# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)

#The example below loads the Pima Indians Diabetes dataset and constructs an Learning Vector Quantization 
#(LVQ) model. The varImp is then used to estimate the variable importance, which is printed and plotted. 
#It shows that the glucose, mass and age attributes are the top 3 most important attributes in the dataset 
#and the insulin attribute is the least important.


# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


#Feature Selection
#+++++++++++++++++++
#Automatic feature selection methods can be used to build many models with different subsets of a dataset and identify those 
#attributes that are and are not required to build an accurate model.

#A popular automatic method for feature selection provided by the caret R package is called Recursive Feature Elimination or RFE.



#The example below provides an example of the RFE method on the Pima Indians Diabetes dataset. A RFE algorithm 
#is used on each iteration to evaluate the model. The algorithm is configured to explore all possible subsets 
#of the attributes. All 8 attributes are selected in this example, although in the plot showing the accuracy of 
#the different attribute subset sizes, we can see that just 4 attributes gives almost comparable results.


#Automatically select features using Caret R PackageR
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
data(PimaIndiansDiabetes)
# define the control using a random feature selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
#The predictors are ranked and the less important ones are sequentially eliminated prior to modeling. The goal is to find a subset of predictors that can be used to produce an accurate model.
results <- rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes[,9], sizes=c(1:8), rfeControl=control)

#sizes  :  a numeric vector of integers corresponding to the number of features that should be retained
#metric : a string that specifies what summary metric will be used to select the optimal model. By default, possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification. 


# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))



#Make Predictions (The final step after feature selection is Making Prediction)


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, PimaIndiansDiabetes)
confusionMatrix(predictions, PimaIndiansDiabetes$diabetes)



#___________________________________________________
#Machine Learning in R: Step-By-Step
# Installing the R platform.
# Loading the dataset.
# Summarizing the dataset.
# Visualizing the dataset.
# Evaluating some algorithms.
# Making some predictions.




# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris
names(dataset)

str(dataset)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]




# Now it is time to take a look at the data.
# 
# In this step we are going to take a look at the data a few different ways:
#   
# Dimensions of the dataset.
# Types of the attributes.
# Peek at the data itself.
# Levels of the class attribute.
# Breakdown of the instances in each class.
# Statistical summary of all attributes.




# dimensions of dataset
dim(dataset1)


# list types for each attribute
str(dataset1)


# list the levels for the class
levels(dataset1$Species)




# summarize the class distribution
percentage <- prop.table(table(dataset1$Species)) * 100
cbind(freq=table(dataset1$Species), percentage=percentage)




# split input and output
x <- dataset[,1:4]
y <- dataset[,5]



#Multivariate Plots
?featurePlot
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")  #ellipse package required


# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")


# density plots for each attribute
featurePlot(x=x, y=y, plot="density")


# pairs plot for each attribute
featurePlot(x=x, y=y, plot="pairs")  #just like ellipse plot with ellipse not present, more like scatter plot




# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)  #here we have scales for each variables, unlike normal density plot


# Kappa=  
#   Pr refers to the proportion of actual (a) and expected (e) agreement between the
# classifier and the true values

F_meas()


#How to decide on threshold values in Machine Learning

