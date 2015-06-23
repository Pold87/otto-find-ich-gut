require(caret)

# Read data
data <- read.csv("data/train80.csv")
data$target <- factor(data$target)
data <- data[-1]

test <- read.csv("data/holdout20.csv")
test <- test[-c(1,ncol(test))]

LogLoss <- function (data, lev = NULL, model = NULL) 
{
    probs <- pmax(pmin(as.numeric(data$T), 1 - 1e-15), 1e-15)
    logPreds <- log(probs)        
    log1Preds <- log(1 - probs)
    real <- (as.numeric(data$obs) - 1)
    out <- c(mean(real * logPreds + (1 - real) * log1Preds)) * -1
    names(out) <- c("LogLoss")
    out
}


# Define models with caret

# Control for caret

myControl <- trainControl(method='cv',
                          number = 1,
                          repeats = 3)


tc <- trainControl(method = "repeatedCV", summaryFunction=LogLoss,
                   number = 10, repeats = 1, verboseIter=TRUE, classProbs=TRUE)

model.gbm <- train(target ~ ., data=data,
                 method= 'gbm',
                 trControl = myControl)


model.rf <- train(target ~ ., data=data,
                  method= 'rf',
                  trControl= tc,
                  metric="LogLoss",
                  maximize=False,
                  preProcess=c("pca"),
                  do.trace = TRUE)


model.treebag <- train(target ~ ., data=data,
               method= 'treebag',
               trControl = myControl)


sub <- predict(model, test, type='prob')

write.csv(sub, "submissions/pls.csv")
