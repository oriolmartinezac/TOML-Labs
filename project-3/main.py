from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import plots

# Packages for the for forward_subset_selection
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def forward_subset_selection(data, n_features=5):
    # Check if missing values for the data
    data.isnull().sum()

    X = data.drop(['Sensor_O3', 'date', 'RefSt'], axis=1)
    y = data['RefSt']

    lreg = LinearRegression()
    # FORWARD by R2
    sfs_r2 = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='r2')
    sfs_r2 = sfs_r2.fit(X, y)
    print(list(sfs_r2.k_feature_names_))
    print(list(sfs_r2.subsets_[1]['cv_scores']))
    print(sfs_r2.k_score_)
    print(sklearn.metrics.get_scorer_names())
    sfs_mae = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='neg_mean_absolute_error')
    sfs_mse = sfs(lreg, k_features=n_features, forward=True, verbose=2, scoring='neg_mean_squared_error')


    feat_names = list(sfs_mse.k_feature_names_)

    return feat_names  # Return the best features


if __name__ == "__main__":
    # CLEAN data before plotting (I.E. dates to datetime, big numbers to numeric)
    new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    # plots.plot_sensor_data(new_PR_data_inner)

    # MAKE THE SUBSET SELECTION FORWARD
    best_features = forward_subset_selection(new_PR_data_inner,
                                             5)  # new_PR_data_inner (dataframe), best n_features to return (list)
    #print(best_features)
