# Breast Cancer Wisconsin Diagnostic Dataset from UCI Machine Learning Repository--------------------
# the project predicts the cancer malignancy or otherwise based on observations of biopsy features
# features correspond to properties of cell nuclei, such as size, shape and regularity. 

options(digits = 3)
# load required libraries
library(tidyverse)
library(caret)
library(dslabs)
data(brca)
dim(brca$x) # 569 observations and 30 features

# brca$y: a vector of sample classifications ("B" = benign or "M" = malignant)

# brca$x: a matrix of numeric features describing properties of the shape and 
# size of cell nuclei extracted from biopsy microscope images

# splitting the data to leave out some validation sets which will be used later on to test the model
test_index <- createDataPartition(brca$y, times = 1, p = 0.9, list = FALSE)
split_x <- brca$x[test_index,]
split_y <- brca$y[test_index]
validation_x <- brca$x[-test_index,]
validation_y <- brca$y[-test_index] # the validation set contains 56 observations

# creating the actual training and test sets
test_index2 <- createDataPartition(split_y, times = 1, p = 0.7, list = FALSE)
train_x <- split_x[test_index2,]
train_y <- split_y[test_index2]
test_x <- split_x[-test_index2,]
test_y <- split_y[-test_index2]

# using logistic regression as the response is binary
train_set <- as.data.frame(train_x) %>% mutate(y = factor(as.numeric(train_y=="B"))) %>% relocate(y) # combining the training data and making the response numeric 
test_set <- as.data.frame(test_x) %>% mutate(y = factor(test_y)) %>% relocate(y) # combining the test data

fit_logit <- glm(y ~ ., data = train_set, family = "binomial", control = list(maxit = 50))

p_hat_logit <- predict(fit_logit, newdata = test_set, type = "response")

y_hat <- ifelse(p_hat_logit > 0.5, "B", "M") %>% factor(levels=levels(test_set$y))
confusionMatrix(y_hat, test_set$y) # 0.948 accuracy

# check how to correct the warning message (0 or 1 numerically occurred)

# using decision tree-----------------------------------------------------------
train_set <- as.data.frame(train_x) %>% mutate(y = factor(train_y)) %>% relocate(y) # modifying the response of the training set to be factors with binary levels
fit_rpart <- train(y ~ ., data = train_set, method = "rpart")
y_hat1 <- predict(fit_rpart, test_set)
confusionMatrix(y_hat1, test_set$y) # 0.915 accuracy 

# using knn---------------------------------------------------------------------
control <- trainControl(method = "cv", number = 10, p = .9) # cross-validation
fit_knn <- train(y ~ ., method = "knn", data = train_set,
                 tuneGrid = data.frame(k = seq(3,51,2)), trControl = control)
y_hat2 <- predict(fit_knn, test_set)
confusionMatrix(y_hat2, test_set$y) # 0.928 accuracy

# using random forest 
control <- trainControl(method = "cv", number = 10, p = .9) # cross-validation
fit_rf <- train(y ~ ., method = "rf", data = train_set, trControl = control)
y_hat3 <- predict(fit_rf, test_set)
confusionMatrix(y_hat3, test_set$y) # 0.941 accuracy

# logistic regression has the highest accuracy of all the fitted models
# putting all together as in the chunk of code below,

df <- data.frame(y = test_set$y, pred1 = y_hat1, pred2 = y_hat2,
                 pred3 = y_hat3) # creating a dataframe of all predictions of the test set response y
comb_fit <- train(y ~ ., method = "rf", data = df) # training on the test data

# checking the accuracies of the models with the validation set
validation_set <- as.data.frame(validation_x) %>% mutate(y = factor(validation_y)) %>% relocate(y) # combining the held-out observations
pred <- predict(fit_logit, validation_set, type = "response")
pred <- ifelse(pred > 0.5, "B", "M") %>% factor(levels=levels(validation_set$y))
mean(pred==validation_set$y) # this performed poorly on the validation set so the model was omitted entirely

pred1 <- predict(fit_rpart, validation_set) # for decision tree model 
pred2 <- predict(fit_knn, validation_set)# knn model 
pred3 <- predict(fit_rf, validation_set)# random forest model

df2 <- data.frame(pred1 = pred1, pred2 = pred2, pred3= pred3)
combined_pred <- predict(comb_fit, df2)

# the code chunk tests the accuracy of the individual models on the validation response

mean(pred1==validation_set$y) # 0.893
mean(pred2==validation_set$y) # 0.911
mean(pred3==validation_set$y) # 0.946
mean(combined_pred==validation_set$y) # 0.946

# the random forest and the ensembled predictions have the highest accuracies when testing on the held-out sample


