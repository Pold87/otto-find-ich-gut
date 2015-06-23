require(xgboost)
require(methods)

train = read.csv('data/train80.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('data/holdout20noclasses.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

random_search <- function(n_set, threads){

#param is a list of parameters

# Set necessary parameter
param <- list("set.seed" = 42,
              "objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = threads,
              "silent"=1,

              "max_depth"=25,
              "eta"=0.1,
              "subsample"=0.7,
              "colsample_bytree"= 1,
              "gamma"=2,
              "min_child_weight"=4,
              "max_delta_step"=0)

param_list <- list()

for (i in seq(n_set)){

param$max_depth <- sample(8:13, 1, replace=T)
param$eta <- runif(1, 0.01, 0.1)
param$subsample <- runif(1, 0.7, 1)
param$colsample_bytree <- runif(1, 0.7, 1)
param$gamma <- runif(1 , 0, 1)
param$min_child_weight <- sample(3:10, 1, replace=T)
param$max_delta_step <- sample(0:3, 1, replace=T)
param_list[[i]] <- param

}

return(param_list)

}


pl <- random_search(100, 2)
LS.df <- do.call(rbind.data.frame, pl)

write.csv(LS.df, "parameters.csv")
line="-- New start --"
write(line,file="scores.txt\n",append=TRUE)


for (param in pl) {
    
    print("New start")
    # Train the model
    nround = 250
    bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

    # Make prediction
    pred = predict(bst,x[teind,])
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)

    # Output submission
    pred = format(pred, digits=4,scientific=F) # shrink the size of submission
    pred = data.frame(1:nrow(pred),pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)

    system("python2 cross_for_R.py")

}
