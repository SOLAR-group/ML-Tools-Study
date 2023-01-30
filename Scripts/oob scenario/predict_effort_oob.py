#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from statistics import stdev
import random
from datetime import (datetime, timedelta)
import time
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
# from sklearn import metrics
import pandas as pd
import xlrd

# Global vars used to compute both fit and prediction times
trainingTime = predictionTime = runningTime = 0

current_milli_time = lambda: int(round(time.time() * 1000))


def format_datetime(mills):
    seconds = mills / 1000
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return '{:02d}:{:05.2f}'.format(minutes.__int__(), seconds)


def read_xls_data(filename):
    # Load the xls file and parse the first sheet
    data = pd.ExcelFile(filename).parse(0)
    clean_data = data.drop("ID", axis=1)
    clean_data.rename(columns={'Effort': 'ActualEffort'}, inplace=True)
    return clean_data


def run_model(model_name, data):
    global trainingTime, predictionTime, runningTime

    trainingTime = predictionTime = runningTime = 0

    # random.seed(248)
    predictions = []
    y = data['ActualEffort']
    data = data.drop(['ActualEffort'], axis=1)
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    for train_index, test_index in loo.split(data):

        X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if model_name == "CART":
            cartModel = DecisionTreeClassifier(criterion="gini")
            startTrainingTime = current_milli_time()
            cartModel.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            cartPredY = cartModel.predict(X_test)
            endPredictionTime = current_milli_time()
            predictionTime = predictionTime + (endPredictionTime - endTrainingTime)
            runningTime = runningTime + (endPredictionTime - startTrainingTime)
            predictions.append(cartPredY)
        elif model_name == "LR":
            lrModel = LinearRegression()
            startTrainingTime = current_milli_time()
            lrModel.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            lrPredY = lrModel.predict(X_test)
            endPredictionTime = current_milli_time()
            predictionTime = predictionTime + (endPredictionTime - endTrainingTime)
            runningTime = runningTime + (endPredictionTime - startTrainingTime)
            predictions.append(lrPredY)
        elif model_name == "KNN":
            knnModel = KNeighborsRegressor()
            startTrainingTime = current_milli_time()
            knnModel.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            knnPredY = knnModel.predict(X_test)
            endPredictionTime = current_milli_time()
            predictionTime = predictionTime + (endPredictionTime - endTrainingTime)
            runningTime = runningTime + (endPredictionTime - startTrainingTime)
            predictions.append(knnPredY)
        elif model_name == "SVM":
            svmModel = SVR()
            startTrainingTime = current_milli_time()
            svmModel.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            svmPredY = svmModel.predict(X_test)
            endPredictionTime = current_milli_time()
            predictionTime = predictionTime + (endPredictionTime - endTrainingTime)
            runningTime = runningTime + (endPredictionTime - startTrainingTime)
            predictions.append(svmPredY)
    results = pd.DataFrame({'predictions': predictions})
    results['actual'] = y
    return results


def returnAbsoluteError(actual, preds):
    error_list = []
    zip_object = zip(actual, preds)
    for actual_i, preds_i in zip_object:
        error_list.append(abs(actual_i - preds_i))
    return error_list


def evaluate_model(preds, actual):
    mae_value = mean_absolute_error(actual, preds)
    medae_value = median_absolute_error(actual, preds)
    stdev_value = stdev(returnAbsoluteError(actual, preds))
    result_list = [mae_value, medae_value, stdev_value]
    # results = pd.DataFrame(result_list, columns = ['MAE', 'MedAe', 'StDev'])
    return result_list


#######################################################################################################
################### Run Code #####################

ds_path = "KitchOriginal.xls"
results_path = "Sklearn_KitchOriginal_RQ1.csv"

data = read_xls_data(filename=ds_path)


print("CART")
for index in range(30):
    cart_predictions = run_model("CART", data)
    cart_results = evaluate_model(cart_predictions['predictions'].astype(int), cart_predictions['actual'])
    cart_results.append(format_datetime(runningTime))
    cart_results.append(format_datetime(trainingTime))
    cart_results.append(format_datetime(predictionTime))
    results_table = pd.DataFrame(columns=['MAE', 'MdAe', 'StDev', 'Time', 'TrainTime', 'PredTime'])
    results_table.loc[len(results_table), :] = cart_results
    if index == 0:
        results_table.to_csv(results_path, mode="a", index=False)
    else:
        if index == 29:
            results_table.loc[len(results_table), :] = ["", "", "", "", "", ""]
        results_table.to_csv(results_path, mode="a", index=False, header=False)

print("KNN")
for index in range(30):
    knn_predictions = run_model("KNN", data)
    knn_results = evaluate_model(knn_predictions['predictions'].astype(int), knn_predictions['actual'])
    knn_results.append(format_datetime(runningTime))
    knn_results.append(format_datetime(trainingTime))
    knn_results.append(format_datetime(predictionTime))
    results_table = pd.DataFrame(columns=['MAE', 'MdAe', 'StDev', 'Time', 'TrainTime', 'PredTime'])
    results_table.loc[len(results_table), :] = knn_results
    if index == 0:
        results_table.to_csv(results_path, mode="a", index=False)
    else:
        if index == 29:
            results_table.loc[len(results_table), :] = ["", "", "", "", "", ""]
        results_table.to_csv(results_path, mode="a", index=False, header=False)

print("LR")
for index in range(30):
    lr_predictions = run_model("LR", data)
    lr_results = evaluate_model(lr_predictions['predictions'].astype(int), lr_predictions['actual'])
    lr_results.append(format_datetime(runningTime))
    lr_results.append(format_datetime(trainingTime))
    lr_results.append(format_datetime(predictionTime))
    results_table = pd.DataFrame(columns=['MAE', 'MdAe', 'StDev', 'Time', 'TrainTime', 'PredTime'])
    results_table.loc[len(results_table), :] = lr_results
    if index == 0:
        results_table.to_csv(results_path, mode="a", index=False)
    else:
        if index == 29:
            results_table.loc[len(results_table), :] = ["", "", "", "", "", ""]
        results_table.to_csv(results_path, mode="a", index=False, header=False)

print("SVM")
for index in range(30):
    svm_predictions = run_model("SVM", data)
    svm_results = evaluate_model(svm_predictions['predictions'].astype(int), svm_predictions['actual'])
    svm_results.append(format_datetime(runningTime))
    svm_results.append(format_datetime(trainingTime))
    svm_results.append(format_datetime(predictionTime))
    results_table = pd.DataFrame(columns=['MAE', 'MdAe', 'StDev', 'Time', 'TrainTime', 'PredTime'])
    results_table.loc[len(results_table), :] = svm_results
    if index == 0:
        results_table.to_csv(results_path, mode="a", index=False)
    else:
        results_table.to_csv(results_path, mode="a", index=False, header=False)

