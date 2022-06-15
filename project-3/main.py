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

# Packages for Ridge Regression
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_regression


def normalize_data(data):
    return (data - data.mean()) / \
           data.std()

def forward_subset_selection(X_train, y_train, n_features=3):
    # Check if missing values for the data
    lreg = LinearRegression()

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
    new_PR_data_inner['date'] = pd.to_datetime(new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    new_PR_data_inner['date'] = new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    new_PR_data_inner['Sensor_O3'] = new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    new_PR_data_inner['Sensor_O3'] = pd.to_numeric(new_PR_data_inner['Sensor_O3'])

    # Create all the plots
    plots.plot_sensor_data(new_PR_data_inner)

    ####### EXERCISE 2 #######

    # Normalize all data
    normalized = normalize_data(new_PR_data_inner[new_PR_data_inner.columns[1:-2]])

    X = normalized.drop(['RefSt'], axis=1)
    y = normalized['RefSt']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # MAKE THE SUBSET SELECTION FORWARD
    best_features = forward_subset_selection(X_train, y_train, 3)  # new_PR_data_inner (dataframe), best n_features to return (list)
    print("\n")
    print(best_features)

    #Copy all the best features in a new dataframe

    # Calculate ridge regression with the features selected
    regression_model = LinearRegression().fit(X_train, y_train)

    ridge_model = linear_model.Ridge()

    n_alphas = 200
    alphas = np.linspace(1, 100, num=20, dtype=int)
    coefs = []
    errors = []

    for a in alphas:
        ridge_model.set_params(alpha=a)
        ridge_model.fit(X_train, y_train)
        coefs.append(ridge_model.coef_)
        pred_ridge_model = ridge_model.predict(X_test)
        print("ALPHA VALUE: ", a)
        print("R²:", metrics.r2_score(y_test, pred_ridge_model))
        print("RMSE: ", metrics.mean_squared_error(y_test, pred_ridge_model, squared=True))
        print("MAE: ", metrics.mean_absolute_error(y_test, pred_ridge_model))


    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')

    plt.show()

    exit()
    """ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)
    print("Ridge model coef {}".format(ridge_model.coef_))
    ridge_model1 = Ridge(alpha=1).fit(X_train, y_train)
    ridge_model5 = Ridge(alpha=5).fit(X_train, y_train)
    ridge_model10 = Ridge(alpha=10).fit(X_train, y_train)

    plt.plot(ridge_model.coef_[0], "s", label="Ridge alpha 0.3")
    plt.plot(ridge_model1.coef_[0], "^", label="Ridge alpha 1")
    plt.plot(ridge_model5.coef_[0], "v", label="Ridge alpha 5")
    plt.plot(ridge_model10.coef_[0], "o", label="Ridge alpha 10")
    plt.plot(regression_model.coef_[0], "o", label="Linear")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, regression_model.coef_[0].itemsize)
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    print("***********************")
    print("RIDGE SCORE ALPHA 0.1")
    print(ridge_model.score(X_train, y_train))
    print(ridge_model.score(X_test, y_test))
    print("***********************")
    print("RIDGE SCORE ALPHA 1")
    print(ridge_model1.score(X_train, y_train))
    print(ridge_model1.score(X_test, y_test))
    print("***********************")
    print("RIDGE SCORE ALPHA 5")
    print(ridge_model5.score(X_train, y_train))
    print(ridge_model5.score(X_test, y_test))
    print("***********************")
    print("RIDGE SCORE ALPHA 10")
    print(ridge_model10.score(X_train, y_train))
    print(ridge_model10.score(X_test, y_test))"""


