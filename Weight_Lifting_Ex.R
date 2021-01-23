# People regularly quantify how much of a particular activity they do, 
# but they rarely quantify how well they do it. In this project, the goal is to use data from 
# accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform 
# barbell lifts correctly and incorrectly in 5 different ways. More information is available here http://groupware.les.inf.puc-rio.br/har 
# under the section on the Weight Lifting Exercise Dataset. 

library(tidyverse)
library(caret)
library(corrplot)
train_set <- read.csv("pml-training.csv") # original training data 
validation <- read.csv("pml-testing.csv") # original test data which serves as an hold-out sample on which the 
                                        # accuracy of the prediction model will be based

y <- train_set$classe
test_index <- createDataPartition(y, times = 1, p = 0.7, list = FALSE) # partitioning the training data
training <- train_set[test_index, ] # creating training set with 70% of the training data 
testing <- train_set[-test_index, ] # similarly, creating a test set 


# Data preprocessing -----------------------------------------------------------
# this stage involves observing and making valuable plots to gain insight into the nature of the predictors
str(training) # shows there 13737 observations with 160 variables
str(testing) # similarly, there 5885 observations with 160 variables

# the str() function also shows there are a lot of NAs which are not useful for prediction purposes
# Columns which are predominantly NAs will be removed as part of the preprocessing stage
# firstly, predictors with weak variability will be removed from the both the training and test set 
# the nearZeroVar() helps perform this function in R by flagging problematic columns


nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]

str(training) # now there are 107 predictors with the same number of observations
str(testing) # similarly, there are now 107 predictors with the number of observations unchanged

# the code chunk below removes predictors which are almost entirely NAs

isna <- sapply(training, function(x) mean(is.na(x))) # this function is the proportion of NAs in each of the columns
table(isna) # this shows that 48 of the 107 predictors have NA proportions greater than 97%
            # the remaining 59 columns do not have any missing value and these are retained
training <- training[, isna == 0]
testing  <- testing[, isna == 0]

dim(training) # 59 columns like we want 
dim(testing) # also has 59 columns

# The chunk below removes the first 5 columns of both the training and test data as they are only for identification
training <- training[,-c(1:5)]
testing <- testing[,-c(1:5)]

# checking for highly correlated predictors, we see there aren't too many highly correlated predictors 
training_cor <-  cor(training[,-54]) # removing the classe column
corrplot(training_cor, method = "color", type = "lower", order = "hclust",
         tl.col = "black", tl.srt = 45, tl.cex = 0.65)


# Prediction methods
# using decision tree-----------------------------------------------------------
training$classe <- factor(training$classe)
testing$classe <- factor(testing$classe)
train_rpart <- train(classe ~ ., method = "rpart", 
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)), data = training)

ggplot(train_rpart, highlight = TRUE) # a plot of the complexity parameter (cp) for visual sake to see the optimal value
train_rpart$bestTune # this confirms it as 0
# access the best model
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75) 

pred <- predict(train_rpart, testing)
confusionMatrix(pred, testing$classe) # accuracy of 0.899

# following this up with a random forest prediction-----------------------------
control <- trainControl(method = "cv", number = 10, p = .9) # using cross-validation
train_rf <- train(classe ~ ., method = "rf", data = training, trControl = control)
pred1 <- predict(train_rf, testing)
confusionMatrix(pred1, testing$classe) # 0.989 accuracy

# Using knn
train_knn <- train(classe ~ ., method = "knn", data = training)
pred2 <- predict(train_knn, testing)
confusionMatrix(pred2, testing$classe) # 0.915 accuracy

# Ensembling using a combination of the different methods
ensemble_df <- data.frame(pred = pred, pred1 = pred1, pred2 = pred2, classe = testing$classe)
ensemble_fit <- train(classe ~ ., data = ensemble_df, method = "rf") # the accuracy of the fit will be checked 
                                                                      # using the held out data
# Applying each of the fitted model to the held-out sample with 20 observations
predv <- predict(train_rpart, validation) # using the decision tree
predv1 <- predict(train_rf, validation) # using random forest
predv2 <- predict(train_knn, validation) # using knn

ens_df <- data.frame(pred = predv, pred1 = predv1, pred2 = predv2)
ensemble_pred <- predict(ensemble_fit, ens_df) # ensemble prediction using a combination of the individual models

# it turns out that the ensembled prediction gives identical prediction of the validation set as the
# random forest. This is shown below by comparing both predictions
mean(predv1==ensemble_pred) # gives 1

# the random forest with an accuracy of 98.9% will, therefore, be use to predict the response of the validation data
# predv1 gives B A B A A E D B A A B C B A E E A B B B

