from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import utilities
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import math
import seaborn as sns

mpl.rcParams['figure.figsize'] = (10, 6)

# Kernel Ridge Regression path to save all the plots
path_kernel_ridge_regression_plots = "./images/kernel_ridge_regression/"


def polynomial_kernel(x, y):
    # divide dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    pred_test = pd.DataFrame()
    pred_test['RefSt'] = y_test
    pred_test['Sensor_O3'] = X_test['Sensor_O3']
    pred_test['date'] = new_PR_data_inner['date']

    errors_rmse = []
    errors_R2 = []
    errors_mae = []

    degree = [i for i in range(1, 26)]
    for d in degree:
        kernel_poly = KernelRidge(kernel='poly', degree=d).fit(X_train, y_train)
        pred_kernel_poly_model = kernel_poly.predict(X_test)
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
    utilities.table_creation(['Degree value', 'R²', 'RMSE', 'MAE'],
                             [degree, errors_R2, errors_rmse, errors_mae],
                             'kernel_ridge_regression_poly.txt')  # Parameters: headers (list), data (list), file (
    # string)


def gaussian_kernel(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = new_PR_data_inner['date']

    alphas = np.arange(0.1, 1.1, 0.1)

    r2 = []
    mse = []
    mae = []

    for a in alphas:
        clf = KernelRidge(kernel='rbf', alpha=a)
        clf.fit(x_train, y_train)
        r2_scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='r2')
        mae_score = cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
        mse_score = cross_val_score(clf, x_train, y_train, cv=10)
        r2.append(r2_scores.mean())
        mse.append(mse_score.mean())
        mae.append(mae_score.mean())

    rmse = [math.sqrt(1 - x) for x in mse]
    mae = [-1 * x for x in mae]
    print(r2)
    print(rmse)
    utilities.table_creation(['Alpha Values', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae],
                             'kernel_ridge_regression_gaussian.txt')

    # plot errors
    plt.title("R-squared")
    plt.xlabel('Lambda')
    plt.ylabel('R^2')
    plt.plot(alphas, r2, color='red')
    plt.savefig(path_kernel_ridge_regression_plots + "error_metrics/kr_rbf_r2.png")
    plt.clf()

    plt.title("Root Mean Squared Error")
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    plt.plot(alphas, rmse, color='blue')
    plt.savefig(path_kernel_ridge_regression_plots + "error_metrics/kr_rbf_rmse.png")
    plt.clf()

    plt.title("Mean Absoulte Error")
    plt.xlabel('Lambda')
    plt.ylabel('MAE')
    plt.plot(alphas, mae, color='black')
    plt.savefig(path_kernel_ridge_regression_plots + "error_metrics/kr_rbf_mae.png")
    plt.clf()

    # best alpha is the one with higher R^2
    best_n = alphas[r2.index(max(r2))]
    print(best_n)

    model = KernelRidge(alpha=best_n)
    model.fit(x_train, y_train)
    mpred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    pred['Pred'] = mpred

    print("RBF PREDICTION")
    print("R^2: ", metrics.r2_score(y_test, mpred))
    print("RMSE: ", metrics.mean_squared_error(y_test, mpred, squared=False))
    print("MAE: ", metrics.mean_absolute_error(y_test, mpred))
    print("Accuracy: ", (acc * 100))

    # Plots
    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='RBF for alpha ' + str(best_n), color='blue')
    plt.savefig(path_kernel_ridge_regression_plots + "models/kr_rbf_pred.png")
    plt.clf()

    sns_rf = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('RBF for alpha' + str(best_n))
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.savefig(path_kernel_ridge_regression_plots + "models/kr_rbf_line.png")
    plt.clf()
