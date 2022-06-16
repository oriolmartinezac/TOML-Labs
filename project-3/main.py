from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import plots
from tabulate import tabulate

import seaborn as sns  # for scatter plot

# Packages to do forward_subset_selection
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Packages to do training and data split
from sklearn.model_selection import train_test_split

# Packages to do Ridge Regression
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_regression


def table_creation(headers, data, file):
    table = {}
    for i, h in enumerate(headers):
        table.update({h: data[i]})
    with open('./tables/' + file, 'w') as file:
        file.write(tabulate(table, headers='keys', tablefmt='fancy_grid'))
    return True


def normalize_data(data):
    return (data - data.mean()) / \
           data.std()


def forward_subset_selection(X_train, y_train, n_features=3):
    # Check if missing values for the data
    lreg = linear_model.LinearRegression()

    # FORWARD by R2
    sfs_r2 = sfs(lreg, k_features=n_features, forward=True, scoring='r2').fit(X_train, y_train)
    sfs_mae = sfs(lreg, k_features=n_features, forward=True, scoring='neg_mean_absolute_error').fit(X_train,
                                                                                                    y_train)
    sfs_mse = sfs(lreg, k_features=n_features, forward=True, scoring='neg_mean_squared_error').fit(X_train,
                                                                                                   y_train)

    feat_names = list(sfs_r2.k_feature_names_)

    """print("\n")
    print("FEATURES                               R^2                   RMSE                     MAE")
    for i, n in enumerate(X.columns):
        print(str(sfs_r2.subsets_[i+1]['feature_names'])+"          "+str(sfs_r2.subsets_[i+1]['cv_scores']))
    """
    return feat_names  # Return the best features


if __name__ == "__main__":
    ####### EXERCISE 1 #######
    # CLEAN data before plotting (I.E. dates to datetime, big numbers to numeric)
    #new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    #new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    #plots.plot_sensor_data(new_PR_data_inner)

    ####### EXERCISE 2 #######

    # Normalize all data
    #normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:-2]]) # WITH PLOTS
    normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:]]) # NO PLOTS

    X = normalized.drop(['RefSt'], axis=1)
    y = normalized['RefSt']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # MAKE THE SUBSET SELECTION FORWARD
    best_features = forward_subset_selection(X_train, y_train,
                                             3)  # new_PR_data_inner (dataframe), best n_features to return (list)

    print("\n")
    print(best_features)

    # Calculate ridge regression with the features selected
    ridge_model = linear_model.Ridge()

    n_alphas = 50
    #alphas = np.linspace(1, 250, num=n_alphas, dtype=int)
    alphas = [1, 5, 10, 100, 500, 1000, 1000000]
    coefs = []
    errors_rmse = []
    errors_R2 = []
    errors_mae = []
    predictions = []
    intercepts = []

    pred_test = pd.DataFrame()
    pred_test['RefSt'] = y_test
    pred_test['Sensor_O3'] = X_test['Sensor_O3']
    pred_test['date'] = new_PR_data_inner['date']

    for a in alphas:
        ridge_model.set_params(alpha=a)
        ridge_model.fit(X_train, y_train)

        print("ALPHA VALUE: ", a)
        print("COEF: ", ridge_model.coef_)
        coefs.append(ridge_model.coef_)
        pred_ridge_model = ridge_model.predict(X_test)
        intercepts.append(ridge_model.intercept_)
        predictions.append(ridge_model.intercept_ + ridge_model.coef_[0] * X_test[X_test.columns[0]] + ridge_model.coef_[1] * X_test[X_test.columns[1]] + ridge_model.coef_[3] * X_test[X_test.columns[3]] + ridge_model.coef_[4] * X_test[X_test.columns[4]] + ridge_model.coef_[5] * X_test[X_test.columns[5]])
        print("R²:", metrics.r2_score(y_test, pred_ridge_model))
        errors_R2.append(metrics.r2_score(y_test, pred_ridge_model))
        print("RMSE: ", metrics.mean_squared_error(y_test, pred_ridge_model, squared=False))
        errors_rmse.append(metrics.mean_squared_error(y_test, pred_ridge_model, squared=False))
        print("MAE: ", metrics.mean_absolute_error(y_test, pred_ridge_model))
        errors_mae.append(metrics.mean_absolute_error(y_test, pred_ridge_model))
        pred_test['Ridge_Pred'] = ridge_model.intercept_ + ridge_model.coef_[0] * X_test['Sensor_O3'] + ridge_model.coef_[1] * X_test['Temp'] + ridge_model.coef_[2] * X_test['RelHum'] + ridge_model.coef_[3] * X_test['Sensor_NO2'] + ridge_model.coef_[4] * X_test['Sensor_NO'] + ridge_model.coef_[5] * X_test['Sensor_SO2']

        ax = pred_test.plot(x='date', y='RefSt')
        pred_test.plot(x='date', y='Ridge_Pred', ax=ax, title='Ridge Regression with alpha=' + str(a))
        plt.show()
        sns_p = sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred_test, fit_reg=True, line_kws={'color': 'orange'}).set(title = 'Ridge Regression with alpha=' + str(a))
        sns_p.set(ylim=(-2, 3))
        sns_p.set(xlim=(-2, 3))
        plt.show()


    # Create the table and save it to a file
    table_creation(['Alpha Values', 'R²', 'RMSE', 'MAE'], [alphas, errors_R2, errors_rmse, errors_mae],
                   'table_ridge_regression.txt')  # Parameters: headers (list), data (list), file (string)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.legend(['Sensor O3', 'Temp', 'RelHum', 'Sensor_NO2', 'Sensor_NO', 'Sensor_SO2'])
    plt.show()

    plt.title("Root Mean Sqare Error (black),  Mean Absolute Error (green), R² (red)")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, errors_rmse, color='black')
    plt.plot(alphas, errors_mae, color='green')
    plt.plot(alphas, errors_R2, color='red')
    plt.show()

    pred_test = pd.DataFrame()
    pred_test['RefSt'] = y_test
    """
    for p in predictions:
        # SCATTER PLOT LOW COST SENSOR O3 AGAINST REFST

        pred_test['MLR_PRED'] = p
        pred_test['Sensor_O3'] = new_PR_data_inner['Sensor_O3']
        #ax = new_PR_data_inner.plot.scatter(x='Sensor_O3', y='RefSt', color='green')

        #plt.plot( , p, ax=ax)

        sns_p = sns.lmplot(x='RefSt', y='MLR_PRED', data=pred_test, fit_reg=True, line_kws={'color': 'orange'})
        sns_p.set(ylim=(-3, 3))
        sns_p.set(xlim=(-3, 5))
        plt.show()"""