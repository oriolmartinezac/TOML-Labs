from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams["figure.figsize"] = (15, 10)
from datetime import datetime

import plots
import utilities

# Packages to do forward_subset_selection
import pandas as pd

# Packages to do training and data split
from sklearn.model_selection import train_test_split

# Packages to do all MLR (Forward Subset Selection, Ridge Regression, Lasso Regression)
import MLR

# Packages to do Kernel Ridge Regression
import KR_RBF

# Packages to do KNN
import KNN

# Packages to do Random Forest
import RF

# Packages to do SVR
import SVR

if __name__ == "__main__":
    ####### PRE-EXERCISE 1 #######
    # CLEAN data before plotting (I.E. dates to datetime, big numbers to numeric)
    # new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    # new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    # plots.plot_sensor_data(new_PR_data_inner)

    ####### EXERCISE 1 #######

    # Normalize all data
    # normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:-2]]) # WITH PLOTS
    normalized = utilities.normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:]])  # NO PLOTS

    X = normalized.drop(['RefSt'], axis=1)
    y = normalized['RefSt']

    # MAKE THE SUBSET SELECTION FORWARD
    MLR.forward_subset_selection(X, y, 3)

    ####### EXERCISE 3 #######
    # Ridge Regression
    MLR.ridge_regression(X, y)

    # Lasso Regression
    MLR.lasso_regression(X, y)

    ####### EXERCISE 3 #######
    # KNN
    KNN.k_neighbors(X, y)

    ####### EXERCISE 4 #######
    # new_X_train = X[['Sensor_O3', 'Temp', 'RelHum']]
    # new_X_test = X_test[['Sensor_O3', 'Temp', 'RelHum']]

    # Kernel Ridge Regression (gaussian function)
    KR_RBF.gaussian_kernel(X, y)

    ####### EXERCISE 5 #######
    # Random Forest
    RF.random_forest(X, y)

    ####### EXERCISE 6 #######
    SVR.svr(X, y)



