from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import pandas as pd
# Packages to do Ridge Regression and Lasso Regression and Forward subset selection
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn import metrics
import matplotlib.pyplot as plt
import utilities
import numpy as np
import seaborn as sns  # for scatter plot
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


from sklearn.model_selection import train_test_split


# FORWARD SUBSET SELECTION path to plots
path_forward_selection_plots = "./images/forward_subset_selection/"

# RIDGE REGRESSION path to plots
path_ridge_regression_plots = "./images/ridge_regression/"

# LASSO REGRESSION path to plot
path_lasso_regression_plots = "./images/lasso_regression/"

def forward_subset_selection(X_train, X_test, y_train, y_test, n_features=3):

    lreg = linear_model.LinearRegression()

    sfs_r2 = sfs(lreg, k_features=n_features, forward=True, scoring='r2').fit(X_train, y_train)
    sfs_mae = sfs(lreg, k_features=n_features, forward=True, scoring='neg_mean_absolute_error').fit(X_train,
                                                                                                    y_train)
    sfs_mse = sfs(lreg, k_features=n_features, forward=True, scoring='neg_mean_squared_error').fit(X_train,
                                                                                                   y_train)
    best_features = list(sfs_r2.k_feature_names_)
    print("Best features", best_features[:])

    best_x_train = X_train[best_features]
    best_x_test = X_test[best_features]

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = X_test['Sensor_O3']
    pred['date'] = new_PR_data_inner['date']

    model = linear_model.LinearRegression()
    model.fit(best_x_train, y_train)
    mpred = model.predict(best_x_test)
    pred["MLR_Pred"] = mpred

    # compute errors
    print("BEST SUBSET PREDICTION")
    r2 = metrics.r2_score(y_test, mpred)
    print("R^2: ", str(r2))
    rmse = metrics.mean_squared_error(y_test, mpred, squared=False)
    print("RMSE: ", str(rmse))
    mae = metrics.mean_absolute_error(y_test, mpred)
    print("MAE: ", str(mae))

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='MLR_Pred', ax=ax1, title='Forward subset selection for ' + str(best_features) + ' features.',
              color='blue')
    # plt.show()
    plt.savefig(path_forward_selection_plots+"models/MLR_forward.png")
    plt.clf()

    sns_rf = sns.lmplot(x='RefSt', y='MLR_Pred', data=pred, fit_reg=True,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('Forward subset selection ' + str(best_features) + ' features.')
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.show()
    plt.savefig(path_forward_selection_plots+"models/MLR_forward_line.png")
    # plt.clf()


def ridge_regression(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

    # Calculate ridge regression with the features selected
    ridge_model = linear_model.Ridge()

    alphas = [0.00001, 0.5, 1, 5, 25, 100, 500, 10000]
    alphas = [0, 1, 5, 10, 50, 100, 250, 500]
    coef = []
    r2 = []
    rmse = []
    mae = []
    for a in alphas:
        rm = linear_model.Ridge()
        rm.set_params(alpha=a)
        rm.fit(x_val, y_val)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Intercept: ", rm.intercept_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(x_test)
        r2.append(metrics.r2_score(y_test, pred))
        rmse.append(metrics.mean_squared_error(y_test, pred, squared=False))
        mae.append(metrics.mean_absolute_error(y_test, pred))

    utilities.table_creation(['Alpha Values', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae], 'ridge_regression_errors.txt')
    utilities.table_creation(['Alpha Values', 'coefs'], [alphas, coef], 'ridge_regression_coefs.txt')

    # plot errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.savefig(path_ridge_regression_plots+"error_metrics/ridge_errors.png")
    plt.clf()
    # plt.show()

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"))
    plt.savefig(path_ridge_regression_plots+"models/ridge_coefs.png")
    plt.clf()
    # plt.show()

    # best alpha is the one with less R^2
    min_r2 = max(r2)
    best_a = alphas[r2.index(min_r2)]

    rm = linear_model.Ridge()
    rm.set_params(alpha=1)
    rm.fit(x_train, y_train)
    rmpred = rm.predict(x_test)
    acc = rm.score(x_test, y_test)

    print("RIDGE PREDICTION")
    print("R^2: ", str(metrics.r2_score(y_test, rmpred)))
    print("RMSE: ", str(metrics.mean_squared_error(y_test, rmpred, squared=False)))
    print("MAE : ", str(metrics.mean_absolute_error(y_test, rmpred)))

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Ridge_Pred'] = rmpred
    pred['date'] = new_PR_data_inner['date']

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Ridge_Pred', ax=ax1, title='Ridge regression for alpha 1', color='blue')
    label = "Ridge_1.png"
    plt.savefig(path_ridge_regression_plots + "models/" + label)
    plt.clf()

    sns_r = sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                       line_kws={'color': 'orange'})
    sns_r.fig.suptitle('Ridge regression for alpha 1')
    sns_r.set(ylim=(-2, 3))
    sns_r.set(xlim=(-2, 3))

    # plt.show()
    label = "Ridge_line_1.png"
    plt.savefig(path_ridge_regression_plots + "models/" + label)
    plt.clf()


def lasso_regression(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

    alphas = np.arange(0.1, 1, 0.1)
    coef = []
    r2 = []
    rmse = []
    mae = []

    for a in alphas:
        rm = linear_model.Lasso()
        rm.set_params(alpha=a)
        rm.fit(x_val, y_val)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Intercept: ", rm.intercept_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(x_test)
        r2.append(metrics.r2_score(y_test, pred))
        rmse.append(metrics.mean_squared_error(y_test, pred, squared=False))
        mae.append(metrics.mean_absolute_error(y_test, pred))

    utilities.table_creation(['Alpha values', 'R²', 'RMSE', 'MAE'], [alphas, r2, rmse, mae], 'lasso_errors.txt')
    utilities.table_creation(['Alpha', 'coefs'], [alphas, coef], 'lasso_coefs.txt')

    # PLOTS
    # Errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.savefig(path_lasso_regression_plots+"error_metrics/lasso_errors.png")
    plt.clf()
    # plt.show()

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"))
    plt.savefig(path_lasso_regression_plots+"models/lasso_coefs.png")
    plt.clf()
    # plt.show()

    # best alpha is the one with less R^2
    min_r2 = max(r2)
    best_a = alphas[r2.index(min_r2)]

    rm = linear_model.Lasso()
    rm.set_params(alpha=best_a)
    rm.fit(x_train, y_train)
    rmpred = rm.predict(x_test)

    print("LASSO PREDICTION")
    print("R^2: ", str(metrics.r2_score(y_test, rmpred)))
    print("RMSE: ", str(metrics.mean_squared_error(y_test, rmpred, squared=False)))
    print("MAE : ", str(metrics.mean_absolute_error(y_test, rmpred)))

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Pred'] = rmpred
    pred['date'] = new_PR_data_inner['date']

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='LASSO for alpha ' + str(best_a), color='blue')
    plt.savefig(path_lasso_regression_plots+"models/lasso_pred.png")
    plt.clf()

    sns_r = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                       line_kws={'color': 'orange'})
    sns_r.fig.suptitle('LASSO for alpha ' + str(best_a))
    sns_r.set(ylim=(-2, 3))
    sns_r.set(xlim=(-2, 3))
    # plt.show()
    plt.savefig(path_lasso_regression_plots+"models/lasso_line.png")
    plt.clf()




