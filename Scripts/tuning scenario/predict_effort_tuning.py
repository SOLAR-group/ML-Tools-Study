from statistics import stdev
from datetime import (datetime, timedelta)
from sklearn.metrics import mean_absolute_error, median_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold

import pandas as pd
import time


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

    # SET 3-FOLDS CV
    kFold = KFold(n_splits=3)

    predictions = []
    y = data['ActualEffort']
    data = data.drop(['ActualEffort'], axis=1)
    loo = LeaveOneOut()
    loo.get_n_splits(data)
    for train_index, test_index in loo.split(data):
        X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if model_name == "CART":
            # SET MAX_DEPTH PARAM
            cart_params = {'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
            cartModel = DecisionTreeClassifier(criterion="gini", random_state=248)
            clf = GridSearchCV(cartModel, param_grid=cart_params, scoring="neg_mean_absolute_error", cv=kFold)
            startTrainingTime = current_milli_time()
            clf.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            cartPredY = clf.predict(X_test)
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
            # SET K PARAM
            knn_params = {'n_neighbors': [1, 5, 9, 13, 17]}
            knnModel = KNeighborsRegressor()
            clf = GridSearchCV(knnModel, param_grid=knn_params, scoring="neg_mean_absolute_error", cv=kFold)
            startTrainingTime = current_milli_time()
            clf.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            knnPredY = clf.predict(X_test)
            endPredictionTime = current_milli_time()
            predictionTime = predictionTime + (endPredictionTime - endTrainingTime)
            runningTime = runningTime + (endPredictionTime - startTrainingTime)
            predictions.append(knnPredY)
        elif model_name == "SVM":
            # SET C AND GAMMA PARAMS
            svm_params = {'C': [0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75, 2, 2.25, 2.50, 2.75, 3, 3.25, 3.50, 3.75, 4],
                          'gamma': [0.1, 0.3, 0.5, 0.7, 0.9]}
            svmModel = SVR(gamma="auto")
            clf = GridSearchCV(svmModel, param_grid=svm_params, scoring="neg_mean_absolute_error", cv=kFold)
            startTrainingTime = current_milli_time()
            clf.fit(X_train, y_train)
            endTrainingTime = current_milli_time()
            trainingTime = trainingTime + (endTrainingTime - startTrainingTime)
            svmPredY = clf.predict(X_test)
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
    return result_list


#######################################################################################################
################### Run Code #####################

ds_path = "KitchDev.csv"
results_path = "Sklearn_KitchDev_RQ2.3.csv"

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

