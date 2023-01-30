library(caret)
library(Metrics)
library(mltools)
library(stats)
library(lubridate)
library(gdata)

trainingTime <- 0.0
predictionTime <- 0.0
runningTime <- 0.0

format_date <- function(seconds) {
  return(strftime(as.POSIXct(seconds_to_period(seconds), origin = Sys.Date()), format = "%M:%OS2"))
}


mae_score <- function(data, lev = NULL, model = NULL) {
  mae_val <- mae(actual = data$obs, predicted = data$pred)
  c(MAE = mae_val)
}


read_xls_data <- function (file_name) {
  data = read.xls(file_name)
  names(data)[names(data) == "Effort"] <- "ActualEffort"
  clean_data = subset(data, select = -ID)
  return(clean_data)
}

run_model <- function(model_name, data) {
  assign("trainingTime", 0.0, envir = .GlobalEnv)
  assign("predictionTime", 0.0, envir = .GlobalEnv)
  assign("runningTime", 0.0, envir = .GlobalEnv)
  # set.seed(248)
  class_label = data$ActualEffort
  predictions = c()
  ## Splitting data accoring to LOO (Leave One Out)
  for (index in 1:nrow(data)) {
    testX = data[index,]
    trainX = data[-index,]
    
    trainY = trainX$ActualEffort
    testY = testX$ActualEffort
    
    training_set = subset(trainX, select = -c(ActualEffort))
    test_set = subset(testX, select = -c(ActualEffort))
    
    if (model_name == "CART") {
      startTrainTime <- Sys.time()
      cartGrid = expand.grid(maxdepth=c(1,3,5,7,9,11,13,15,17,19,21))
      cartModel <- train(x = training_set, y = trainY, method = "rpart2", control=list(cp=0), metric = "MAE", tuneGrid = cartGrid ,
                         trControl = trainControl(method = "cv", number = 3, summaryFunction = mae_score))
      endTrainTime <- Sys.time()
      assign("trainingTime", get("trainingTime", envir = .GlobalEnv) + as.numeric(endTrainTime - startTrainTime), envir = .GlobalEnv)
      cartPredY = predict(cartModel, test_set)
      endPredictionTime <- Sys.time()
      assign("predictionTime", get("predictionTime", envir = .GlobalEnv) + as.numeric(endPredictionTime - endTrainTime), envir = .GlobalEnv)
      predictions = c(predictions, cartPredY)
    }
    else if (model_name == "LR") {
      startTrainTime <- Sys.time()
      lrModel <- train(x = training_set, y = trainY, method = "lm")
      endTrainTime <- Sys.time()
      assign("trainingTime", get("trainingTime", envir = .GlobalEnv) + as.numeric(endTrainTime - startTrainTime), envir = .GlobalEnv)
      lrPredY = predict(lrModel, test_set)
      endPredictionTime <- Sys.time()
      assign("predictionTime", get("predictionTime", envir = .GlobalEnv) + as.numeric(endPredictionTime - endTrainTime), envir = .GlobalEnv)
      predictions = c(predictions, lrPredY)
    }
    else if (model_name == "KNN") {
      startTrainTime <- Sys.time()
      knnGrid = expand.grid(k=c(1, 5, 9, 13, 17))
      knnModel <- train(x = training_set, y = trainY, method = "knn",  metric = "MAE", tuneGrid = knnGrid,
                        trControl = trainControl(method = "cv", number = 3, summaryFunction = mae_score))
      endTrainTime <- Sys.time()
      assign("trainingTime", get("trainingTime", envir = .GlobalEnv) + as.numeric(endTrainTime - startTrainTime), envir = .GlobalEnv)
      knnPredY = predict(knnModel, test_set)
      endPredictionTime <- Sys.time()
      assign("predictionTime", get("predictionTime", envir = .GlobalEnv) + as.numeric(endPredictionTime - endTrainTime), envir = .GlobalEnv)
      predictions = c(predictions, knnPredY)
    }
    else if (model_name == "SVM") {
      startTrainTime <- Sys.time()
      svmGrid= expand.grid(C=c(0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75, 2, 2.25, 2.50, 2.75, 3, 3.25, 3.50, 3.75, 4), 
                           sigma= c(0.1, 0.3, 0.5, 0.7, 0.9))
      svmModel <- train(x = training_set, y = trainY, method = "svmRadial", metric = "MAE", tuneGrid = svmGrid,
                        trControl = trainControl(method = "cv", number = 3, summaryFunction = mae_score))
      endTrainTime <- Sys.time()
      assign("trainingTime", get("trainingTime", envir = .GlobalEnv) + as.numeric(endTrainTime - startTrainTime), envir = .GlobalEnv)
      svmPredY = predict(svmModel, test_set)
      endPredictionTime <- Sys.time()
      assign("predictionTime", get("predictionTime", envir = .GlobalEnv) + as.numeric(endPredictionTime - endTrainTime), envir = .GlobalEnv)
      predictions = c(predictions, svmPredY)
    }
  }
  ### combine actual and predictions in one object
  assign("runningTime", get("trainingTime", envir = .GlobalEnv) + get("predictionTime", envir = .GlobalEnv), envir = .GlobalEnv)
  list <- list("predictions" = predictions, "actual" = class_label)
  return(list)
}

evaluate_model <- function(preds, actual) {
  #mean absolute error 
  mae_value = mae(actual, preds)
  
  # median absolute error 
  medae_value = mdae(actual, preds)
  
  # standard deviation 
  stdev_value = sd(ae(actual, preds))
  
  all_metrics = list("mae" = mae_value, "medae" = medae_value, "stdev" = stdev_value)
  
  return(all_metrics)
  ## add medae and stdev 
  
}

#####################################################################################################################

data = read_xls_data("KitchDev.xls")
result_path = "Caret_KitchDev_RQ1.csv"

print("CART")
for(index in 1:30){
  cart_predictions = run_model("CART", data)
  cart_results = evaluate_model(cart_predictions$predictions, cart_predictions$actual)
  cart_list_results = c(cart_results$mae, cart_results$medae, cart_results$stdev, format_date(get("runningTime", envir = .GlobalEnv)), format_date(get("trainingTime", envir = .GlobalEnv)), format_date(get("predictionTime", envir = .GlobalEnv)))
  all_results = data.frame(c(cart_list_results))
  row.names(all_results) = c("MAE", "MdAe", "StDev", "Time", "TrainTime", "PredTime")
  if(index == 1)
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".",  col.names = TRUE, row.names = FALSE)
  else
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".", col.names = FALSE, row.names = FALSE)
}
print("KNN")
for(index in 1:30){
  knn_predictions = run_model("KNN", data)
  knn_results = evaluate_model(knn_predictions$predictions, knn_predictions$actual)
  knn_list_results = c(knn_results$mae, knn_results$medae, knn_results$stdev, format_date(get("runningTime", envir = .GlobalEnv)), format_date(get("trainingTime", envir = .GlobalEnv)), format_date(get("predictionTime", envir = .GlobalEnv)))
  all_results = data.frame(c(knn_list_results))
  row.names(all_results) = c("MAE", "MdAe", "StDev", "Time", "TrainTime", "PredTime")
  if(index == 1)
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".",  col.names = TRUE, row.names = FALSE)
  else
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".", col.names = FALSE, row.names = FALSE)
}
print("LR")
for(index in 1:30){
  lr_predictions = run_model("LR", data)
  lr_results = evaluate_model(lr_predictions$predictions, lr_predictions$actual)
  lr_list_results = c(lr_results$mae, lr_results$medae, lr_results$stdev, format_date(get("runningTime", envir = .GlobalEnv)), format_date(get("trainingTime", envir = .GlobalEnv)), format_date(get("predictionTime", envir = .GlobalEnv)))
  all_results = data.frame(c(lr_list_results))
  row.names(all_results) = c("MAE", "MdAe", "StDev", "Time", "TrainTime", "PredTime")
  if(index == 1)
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".",  col.names = TRUE, row.names = FALSE)
  else
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".", col.names = FALSE, row.names = FALSE)
}
print("SVM")
for(index in 1:30){
  svm_predictions = run_model("SVM", data)
  svm_results = evaluate_model(svm_predictions$predictions, svm_predictions$actual)
  svm_list_results = c(svm_results$mae, svm_results$medae, svm_results$stdev, format_date(get("runningTime", envir = .GlobalEnv)), format_date(get("trainingTime", envir = .GlobalEnv)), format_date(get("predictionTime", envir = .GlobalEnv)))
  all_results = data.frame(c(svm_list_results))
  row.names(all_results) = c("MAE", "MdAe", "StDev", "Time", "TrainTime", "PredTime")
  if(index == 1)
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".",  col.names = TRUE, row.names = FALSE)
  else
    write.table(t(all_results), result_path, append = TRUE, sep = ",", dec = ".", col.names = FALSE, row.names = FALSE)
}






