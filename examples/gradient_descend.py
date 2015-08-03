__author__ = 'NLP-PC'
# coding: utf8
x_old = 0
x_new = 6  # step size
eps = 0.01
precison = 0.00001


def f_derivative(x):
    return 4 * x ** 3 - 9 * x ** 2


while abs(x_new - x_old) > precison:
    x_old = x_new
    x_new = x_old - eps * f_derivative(x_old)

print("Local minimum occurs at: ", x_new)
