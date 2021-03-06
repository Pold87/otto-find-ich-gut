library(h2o)
localH2O <- h2o.init(nthread=2)

train <- read.csv("data/train25.csv")

for(i in 2:94){
train[,i] <- as.numeric(train[,i])
train[,i] <- sqrt(train[,i]+(3/8))
}


test <- read.csv("data/test25.csv")

for(i in 2:94){
test[,i] <- as.numeric(test[,i])
test[,i] <- sqrt(test[,i]+(3/8))
}



train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test[,2:94])

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

submission <- read.csv("IPython_Submissions/xgboostpython_25.csv")
submission[,2:10] <- 0

for(i in 1:3){
print(i)
model <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train.hex,
                          classification=T,
                          activation="RectifierWithDropout",
                          hidden=c(1024,512,256),
                          hidden_dropout_ratio=c(0.5,0.5,0.5),
                          input_dropout_ratio=0.05,
                          epochs=100,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=2000,
                          max_w2=10,
                          seed=1)
submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
print(i)
write.csv(submission,file="h2onewcrossvalidation_100_epochs.csv",row.names=FALSE) 
}      

                   
