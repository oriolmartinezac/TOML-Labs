from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import utilities
from sklearn.model_selection import train_test_split

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


def gaussian_kernel(X_train, X_test, y_train, y_test):
    pred_test = pd.DataFrame()
    pred_test['RefSt'] = y_test
    pred_test['Sensor_O3'] = X_test['Sensor_O3']
    pred_test['date'] = new_PR_data_inner['date']

    errors_rmse = []
    errors_R2 = []
    errors_mae = []

    kernel_rbf = KernelRidge(kernel="rbf").fit(X_train, y_train)

    pred_kernel_gauss_model = kernel_rbf.predict(X_test)
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
    utilities.table_creation(['R²', 'RMSE', 'MAE'],
                             [errors_R2, errors_rmse, errors_mae],
                             'kernel_ridge_regression_gaussian.txt')  # Parameters: headers (list), data (list),
    # file (string)

