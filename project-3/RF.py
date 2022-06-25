from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

import utilities
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

path_random_forest_plots = "./images/random_forest/"


def random_forest(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = new_PR_data_inner['date']

    n_trees = [1, 2, 5, 7, 10, 13, 15, 18, 20, 25]

    r2 = []
    mse = []
    mae = []

    for n in n_trees:
        rf = RandomForestRegressor(n_estimators=n)
        rf.fit(x_train, y_train)
        r2_scores = cross_val_score(rf, x_train, y_train, cv=10, scoring='r2')
        mae_score = cross_val_score(rf, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
        mse_score = cross_val_score(rf, x_train, y_train, cv=10)
        r2.append(r2_scores.mean())
        mse.append(mse_score.mean())
        mae.append(mae_score.mean())

    rmse = [math.sqrt(1 - x) for x in mse]
    mae = [-1 * x for x in mae]
    print(r2)
    print(rmse)
    utilities.table_creation(['Number of trees', 'R^2', 'RMSE', 'MAE'], [n_trees, r2, rmse, mae],
                             'rf_table.txt')

    # plot errors
    plt.title("R-squared")
    plt.xlabel('Number of trees')
    plt.ylabel('R^2')
    plt.plot(n_trees, r2, color='red')
    plt.savefig(path_random_forest_plots + "error_metrics/rf_r2.png", bbox_inches='tight')
    plt.clf()

    plt.title("Root Mean Squared Error")
    plt.xlabel('Number of trees')
    plt.ylabel('RMSE')
    plt.plot(n_trees, rmse, color='blue')
    plt.savefig(path_random_forest_plots + "error_metrics/rf_rmse.png", bbox_inches='tight')
    plt.clf()

    plt.title("Mean Absolute Error")
    plt.xlabel('Number of trees')
    plt.ylabel('MAE')
    plt.plot(n_trees, mae, color='black')
    plt.savefig(path_random_forest_plots + "error_metrics/rf_mae.png", bbox_inches='tight')
    plt.clf()

    # best number of trees is the one with higher R^2
    best_n = n_trees[r2.index(max(r2))]
    print(best_n)

    model = RandomForestRegressor(n_estimators=best_n)
    model.fit(x_train, y_train)
    mpred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    pred['Pred'] = mpred

    print("RF PREDICTION")
    print("R^2: ", metrics.r2_score(y_test, mpred))
    print("RMSE: ", metrics.mean_squared_error(y_test, mpred, squared=False))
    print("MAE: ", metrics.mean_absolute_error(y_test, mpred))
    print("Accuracy: ", (acc * 100))

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='Random Forest for ' + str(best_n) + ' trees.', color='blue')
    label = "RF_" + str(best_n) + ".png"
    plt.savefig(path_random_forest_plots + "models/" + label)
    plt.clf()

    # sns.set(rc={"figure.figsize": (12, 15)})
    sns_rf = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('Random Forest for ' + str(best_n) + ' trees.')
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    label = "RF_line_" + str(best_n) + ".png"
    plt.savefig(path_random_forest_plots + "models/" + label)
    plt.clf()
