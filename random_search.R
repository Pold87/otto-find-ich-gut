require(xgboost)
require(methods)


random_search <- function(n_set, threads){

#param is a list of parameters

# Set necessary parameter
param <- list("set.seed" = 42,
              "objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = threads,
              "silent"=1,
              "max_depth"=10,
              "eta"=0.1,
              "subsample"=0.7,
              "colsample_bytree"= 1,
              "gamma"=2,
              "min_child_weight"=4)

param_list <- list()

for (i in seq(n_set)){

## n_par <- length(param)


param$max_depth <- sample(8:13,1, replace=T)

param$eta <- runif(1,0.01,0.2)
param$subsample <- runif(1,0.7,1)
param$colsample_bytree <- runif(1,0.7,0.95)
param$min_child_weight <- sample(3:17,1, replace=T)
param$gamma <- runif(1,2,10)
param$min_child_weight <- sample(1:15,1, replace=T)
param_list[[i]] <- param

}

return(param_list)

}


print(random_search(20, 2)[0])
