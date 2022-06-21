from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams["figure.figsize"] = (15, 10)
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
from sklearn.linear_model import RidgeCV

# Packages to do Kernel Ridge Regression
from sklearn.kernel_ridge import KernelRidge


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
    # new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    # new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    # plots.plot_sensor_data(new_PR_data_inner)

    ####### EXERCISE 2 #######

    # Normalize all data
    # normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:-2]]) # WITH PLOTS
    normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:]])  # NO PLOTS

    X = normalized.drop(['RefSt'], axis=1)
    y = normalized['RefSt']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # MAKE THE SUBSET SELECTION FORWARD
    best_features = forward_subset_selection(X_train, y_train,
                                             3)  # new_PR_data_inner (dataframe), best n_features to return (list)

    print("\n")
    print(best_features)

    # RIDGE REGRESSION
    path_ridge_regression_plots = "./images/ridge_regression/"

    # Calculate ridge regression with the features selected
    ridge_model = linear_model.Ridge()

    n_alphas = 50
    # alphas = np.linspace(1, 250, num=n_alphas, dtype=int)
    alphas = [0.00001, 0.5, 1, 5, 25, 100, 500, 10000]
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

    clf = RidgeCV(alphas=alphas).fit(X_train, y_train)
    print("BEST ALPHA", clf.alpha_)

    for a in alphas:
        ridge_model.set_params(alpha=a)
        ridge_model.fit(X_train, y_train)

        print("ALPHA VALUE: ", a)
        print("COEF: ", ridge_model.coef_)
        coefs.append(ridge_model.coef_)
        pred_ridge_model = ridge_model.predict(X_test)
        intercepts.append(ridge_model.intercept_)
        predictions.append(
            ridge_model.intercept_ + ridge_model.coef_[0] * X_test[X_test.columns[0]] + ridge_model.coef_[1] * X_test[
                X_test.columns[1]] + ridge_model.coef_[3] * X_test[X_test.columns[3]] + ridge_model.coef_[4] * X_test[
                X_test.columns[4]] + ridge_model.coef_[5] * X_test[X_test.columns[5]])
        print("R²:", metrics.r2_score(y_test, pred_ridge_model))
        errors_R2.append(metrics.r2_score(y_test, pred_ridge_model))
        print("RMSE: ", metrics.mean_squared_error(y_test, pred_ridge_model, squared=False))
        errors_rmse.append(metrics.mean_squared_error(y_test, pred_ridge_model, squared=False))
        print("MAE: ", metrics.mean_absolute_error(y_test, pred_ridge_model))
        errors_mae.append(metrics.mean_absolute_error(y_test, pred_ridge_model))
        pred_test['Ridge_Pred'] = ridge_model.intercept_ + ridge_model.coef_[0] * X_test['Sensor_O3'] + \
                                  ridge_model.coef_[1] * X_test['Temp'] + ridge_model.coef_[2] * X_test['RelHum'] + \
                                  ridge_model.coef_[3] * X_test['Sensor_NO2'] + ridge_model.coef_[4] * X_test[
                                      'Sensor_NO'] + ridge_model.coef_[5] * X_test['Sensor_SO2']

        ax = pred_test.plot(x='date', y='RefSt')
        pred_test.plot(x='date', y='Ridge_Pred', ax=ax, title='Ridge Regression with alpha=' + str(a))
        plt.savefig(path_ridge_regression_plots + str("models/ridge_model_a-" + str(a) + ".png"), bbox_inches='tight')
        plt.clf()

        sns_p = sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred_test, fit_reg=True, line_kws={'color': 'orange'}).set(
            title='Ridge Regression with alpha=' + str(a))
        sns_p.set(ylim=(-2, 3))
        sns_p.set(xlim=(-2, 3))
        sns_p.savefig(path_ridge_regression_plots + str("models/sns_plot_a-" + str(a) + ".png"), bbox_inches='tight')
        plt.clf()

    # Create the table and save it to a file
    table_creation(['Alpha Values', 'R²', 'RMSE', 'MAE', 'COEFS'], [alphas, errors_R2, errors_rmse, errors_mae, coefs],
                   'table_ridge_regression.txt')  # Parameters: headers (list), data (list), file (string)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.legend(['Sensor O3', 'Temp', 'RelHum', 'Sensor_NO2', 'Sensor_NO', 'Sensor_SO2'])
    plt.axis('tight')
    plt.savefig(path_ridge_regression_plots + str("models/all_coefs.png"), bbox_inches='tight')
    plt.clf()

    plt.title("Root Mean Sqare Error (black),  Mean Absolute Error (green), R² (red)")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, errors_rmse, color='black')
    plt.plot(alphas, errors_mae, color='green')
    plt.plot(alphas, errors_R2, color='red')
    plt.savefig(path_ridge_regression_plots + str("error_metrics/all_errors.png"), bbox_inches='tight')
    plt.clf()

    plt.plot(alphas, errors_rmse)
    plt.show()

    plt.plot(alphas, errors_mae)
    plt.show()

    plt.plot(alphas, errors_R2)
    plt.show()

    errors_rmse.clear()
    errors_R2.clear()
    errors_mae.clear()

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

    ####### EXERCISE 4 #######
    new_X_train = X_train[['Sensor_O3', 'Temp', 'RelHum']]
    new_X_test = X_test[['Sensor_O3', 'Temp', 'RelHum']]

    # Kernel Ridge Regression
    path_kernel_ridge_regression_plots = "./images/kernel_ridge_regression/"

    # POLYNOMIAL KERNEL FUNCTION
    degree = [i for i in range(1, 26)]
    for d in degree:
        kernel_poly = KernelRidge(kernel='poly', degree=d).fit(new_X_train, y_train)
        pred_kernel_poly_model = kernel_poly.predict(new_X_test)
        print("R²:", metrics.r2_score(y_test, pred_kernel_poly_model))
        errors_R2.append(metrics.r2_score(y_test, pred_kernel_poly_model))
        print("RMSE: ", metrics.mean_squared_error(y_test, pred_kernel_poly_model, squared=False))
        errors_rmse.append(metrics.mean_squared_error(y_test, pred_kernel_poly_model, squared=False))
        print("MAE: ", metrics.mean_absolute_error(y_test, pred_kernel_poly_model))
        errors_mae.append(metrics.mean_absolute_error(y_test, pred_kernel_poly_model))
        if d == 7:  # Best degree for the kernel function
            pred_test['Best_Kernel_Poly_Pred'] = pred_kernel_poly_model
        else:
            pred_test['Kernel_Poly_Pred'] = pred_kernel_poly_model

        # Plots
        ax = pred_test.plot(x='date', y='RefSt')
        pred_test.plot(x='date', y='Kernel_Poly_Pred', ax=ax,
                       title='Polynomial Kernel Ridge Regression with degree=' + str(d))
        plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_poly_model_d-" + str(d) + ".png"),
                    bbox_inches='tight')
        plt.clf()

    kr_stats = pd.DataFrame({'degree': degree, 'r_squared': errors_R2, 'rmse': errors_rmse, 'mae': errors_mae})
    kr_stats = kr_stats.set_index('degree')  # index column (X axis for the plots)

    kr_stats[["r_squared"]].plot()
    plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/r_squared_polynomial.png"), bbox_inches='tight')
    plt.clf()

    kr_stats[["rmse"]].plot()
    plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/rmse_polynomial.png"), bbox_inches='tight')
    plt.clf()

    kr_stats[["mae"]].plot()
    plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/mae_polynomial.png"), bbox_inches='tight')
    plt.clf()

    # Create the table and save it to a file
    table_creation(['Degree value', 'R²', 'RMSE', 'MAE'],
                   [degree, errors_R2, errors_rmse, errors_mae],
                   'kernel_ridge_regression_poly.txt')  # Parameters: headers (list), data (list), file (string)

    # GAUSSIAN KERNEL FUNCTION
    errors_rmse.clear()
    errors_R2.clear()
    errors_mae.clear()

    kernel_rbf = KernelRidge(kernel="rbf").fit(new_X_train, y_train)

    pred_kernel_gauss_model = kernel_rbf.predict(new_X_test)
    print("R²:", metrics.r2_score(y_test, pred_kernel_gauss_model))
    errors_R2.append(metrics.r2_score(y_test, pred_kernel_gauss_model))
    print("RMSE: ", metrics.mean_squared_error(y_test, pred_kernel_gauss_model, squared=False))
    errors_rmse.append(metrics.mean_squared_error(y_test, pred_kernel_gauss_model, squared=False))
    print("MAE: ", metrics.mean_absolute_error(y_test, pred_kernel_gauss_model))
    errors_mae.append(metrics.mean_absolute_error(y_test, pred_kernel_gauss_model))
    pred_test['Kernel_Gauss_Pred'] = pred_kernel_gauss_model

    # Plot
    ax = pred_test.plot(x='date', y='RefSt')
    pred_test.plot(x='date', y='Kernel_Gauss_Pred', ax=ax,
                   title='Gaussian Kernel Ridge Regression')
    plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_gauss_model.png"),
                bbox_inches='tight')
    plt.clf()

    # Create the table and save it to a file
    table_creation(['R²', 'RMSE', 'MAE'],
                   [errors_R2, errors_rmse, errors_mae],
                   'kernel_ridge_regression_gaussian.txt')  # Parameters: headers (list), data (list), file (string)

    # Both plots
    ax = pred_test.plot(x='date', y='RefSt')
    ax2 = pred_test.plot(x='date', y='Kernel_Gauss_Pred', ax=ax)
    pred_test.plot(x='date', y='Best_Kernel_Poly_Pred', ax=ax,
                   title='Different')
    plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_gauss_poly_model.png"),
                bbox_inches='tight')
    plt.clf()
