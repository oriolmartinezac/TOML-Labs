import pandas as pd
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import utilities
import math
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10, 6)


def k_neighbors(x_train, x_test, y_train, y_test):

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = new_PR_data_inner['date']

    n_neighbors = [2, 5, 8, 10, 12, 15, 18, 22, 25]
    r2 = []
    mse = []
    mae = []
    for k in n_neighbors:  # running for different K values to know which yields the max accuracy.
        clf = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1)
        clf.fit(x_train, y_train)
        r2_scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='r2')
        mae_score = cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
        mse_score = cross_val_score(clf, x_train, y_train, cv=10)
        r2.append(r2_scores.mean())
        mse.append(mse_score.mean())
        mae.append(mae_score.mean())

    rmse = [math.sqrt(1 - x) for x in mse]
    mae = [-1*x for x in mae]
    print(r2)
    print(rmse)
    utilities.table_creation(['Number of neighbours', 'R^2', 'RMSE', 'MAE'], [n_neighbors, r2, rmse, mae],
                         'knn_table.txt')

    # plot errors
    plt.title("R-squared")
    plt.xlabel('Number of neighbors')
    plt.ylabel('R^2')
    plt.plot(n_neighbors, r2, color='red')
    plt.savefig("img/knn_r2")
    plt.clf()

    plt.title("Root Mean Squared Error")
    plt.xlabel('Number of neighbors')
    plt.ylabel('RMSE')
    plt.plot(n_neighbors, rmse, color='blue')
    plt.savefig("img/knn_rmse")
    plt.clf()

    plt.title("Mean Absoulte Error")
    plt.xlabel('Number of neighbors')
    plt.ylabel('MAE')
    plt.plot(n_neighbors, mae, color='black')
    plt.savefig("img/knn_mae")
    plt.clf()
    # plt.show()

    # best neighbour is the one with higher R^2
    best_n = n_neighbors[r2.index(max(r2))]
    print(best_n)

    model = KNeighborsRegressor(n_neighbors=best_n)
    model.fit(x_train, y_train)
    mpred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    pred['Pred'] = mpred

    print("KNN PREDICTION")
    print("R^2: ", str(r2_score(y_test, mpred)))
    print("RMSE: ", str(mean_squared_error(y_test, mpred, squared=False)))
    print("MAE: ", str(mean_absolute_error(y_test, mpred)))
    print("Accuracy: ", str(acc*100))

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='KNN for ' + str(best_n) + ' neighbors.', color='blue')
    label = "KNN_" + str(best_n)
    plt.savefig("img/" + label)
    plt.clf()

    # sns.set(rc={"figure.figsize": (12, 15)})
    sns_rf = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('KNN for ' + str(best_n) + ' neighbors.')
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    label = "KNN_line_" + str(best_n)
    plt.savefig("img/" + label)
    plt.clf()