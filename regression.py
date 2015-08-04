__author__ = 'NLP-PC'
from sklearn import linear_model
from evaluate import evaluate
from visualization import draw_line, draw_linear_regression
import numpy as np


def linear_regression(X_train, X_test, Y_train, Y_test, plot=False):
    # Create linear regression object
    # The training data should be column vectors
    X_train, X_test = np.array(X_train).reshape((len(X_train), 1)), np.array(X_test).reshape((len(X_test), 1))
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    predict = regr.predict(X_test)
    # record the experiment performance, Explained variance score: 1 is perfect prediction
    np.seterr(invalid='ignore')
    evaluate(list(predict), np.array(Y_test),
             'linear regression ' + 'Explained variance score: %.2f' % regr.score(X_test, Y_test))
    if plot is True:
        draw_linear_regression(X_train, Y_train, regr.predict(X_train))
        draw_linear_regression(X_test, Y_test, predict)


def linear_regression_multivariant(X_train, X_test, Y_train, Y_test, cost_fun='ordinary_least_squares'):
    if cost_fun == 'ordinary_least_squares':
        regr = linear_model.LinearRegression()
    elif cost_fun == 'Ridge_Regression':
        regr = linear_model.Ridge(alpha=.5)
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    predict = regr.predict(X_test)
    # record the experiment performance, Explained variance score: 1 is perfect prediction
    np.seterr(invalid='ignore')
    evaluate(list(predict), np.array(Y_test),
             'linear regression ' + 'Explained variance score: %.2f' % regr.score(X_test, Y_test))

if __name__ == '__main__':
    linear_regression_multivariant([[1, 2, 3], [1, 2, 4]], [[5, 6, 7], [2, 3, 5]], [2, 5], [6, 7])
    linear_regression([1, 2, 3], [4, 6], [1, 4, 9], [16, 36], plot=False)
