from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import plots

# Packages for the for forward_subset_selection
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def normalize_data(data):
    return (data - data.mean()) / \
           data.std()


def check_if_null_elements(data):
    data.isnull().sum()


def forward_subset_selection(X_train, y_train, n_features=3):
    # Check if missing values for the data
    lreg = LinearRegression()

    # FORWARD by R2
    sfs_r2 = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='r2').fit(X_train, y_train)
    sfs_mae = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='neg_mean_absolute_error').fit(X_train,
                                                                                                               y_train)
    sfs_mse = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='neg_mean_squared_error').fit(X_train,
                                                                                                              y_train)

    feat_names = list(sfs_r2.k_feature_names_)

    """print("\n")
    print("FEATURES                               R^2                   RMSE                     MAE")
    for i, n in enumerate(X.columns):
        print(str(sfs_r2.subsets_[i+1]['feature_names'])+"          "+str(sfs_r2.subsets_[i+1]['cv_scores']))
    """
    return feat_names  # Return the best features


if __name__ == "__main__":
    # CLEAN data before plotting (I.E. dates to datetime, big numbers to numeric)
    new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    plots.plot_sensor_data(new_PR_data_inner)

    # Check if the elements in the dataframes are null
    check_if_null_elements(new_PR_data_inner)

    X = new_PR_data_inner.drop(['date', 'RefSt'], axis=1)
    y = new_PR_data_inner['RefSt']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=False)


    best_features = forward_subset_selection(X, y, 3)  # new_PR_data_inner (dataframe), best n_features to return (list)

    # Normalize all the best features
    normalized = normalize_data(new_PR_data_inner[best_features[:]])


    # MAKE THE SUBSET SELECTION FORWARD
    # best_features = forward_subset_selection(new_PR_data_inner, 3)
    print("\n")
    print(best_features)
