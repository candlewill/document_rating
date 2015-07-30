__author__ = 'NLP-PC'
import sys
def few(msg):
    print(msg)
    print(sys._getframe().f_code.co_name)


from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(clf.coef_)
