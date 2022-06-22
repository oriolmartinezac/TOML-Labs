from MLR_Build_File_Pandas_HW3 import *  # IMPORTING HEADER FILE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns  # for scatter plot

path_support_vector_regression = "./images/support_vector_regression/"


def svr(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = new_PR_data_inner['date']

    # Performing hyper-parameters grid search
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    # gamma = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gamma = ["auto", "scale"]
    epsilon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    kernel = ["rbf"]

    param_grid = {
        'kernel': kernel,
        'gamma': gamma,
        'C': C,
        'epsilon': epsilon
    }

    scoring_cols = [
        'param_kernel',
        'param_C',
        'param_gamma',
        'param_epsilon',
        'mean_test_mae',
        'mean_test_mse',
        'mean_test_r2',
    ]

    scoring_dict = {
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
    }

    cvModel = GridSearchCV(
        estimator=SVR(),
        scoring=scoring_dict,
        param_grid=param_grid,
        refit='mae',
        cv=10,
        n_jobs=-1,
        return_train_score=False
    )

    cvModel = cvModel.fit(x_train, y_train)

    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    scores = pd.DataFrame(cvModel.cv_results_).sort_values(by='mean_test_mae', ascending=False)[scoring_cols].head()
    print(scores)

    pred['SVR_Pred'] = cvModel.predict(x_test)
    pred['date'] = new_PR_data_inner['date']
    ax = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='SVR_Pred', ax=ax, title='Support Vector Regression', color='blue')
    plt.savefig(path_support_vector_regression + "models/SVR_pred_auto.png")
    plt.clf()
    # plt.show()

    # Plot regression
    sns_svr = sns.lmplot(x='RefSt', y='SVR_Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                         line_kws={'color': 'orange'})
    sns_svr.fig.suptitle('Best prediction for SVR')
    sns_svr.set(ylim=(-2, 3))
    sns_svr.set(xlim=(-2, 3))
    plt.savefig(path_support_vector_regression + "models/SVR_line_auto.png")
    plt.clf()
    # plt.show()
