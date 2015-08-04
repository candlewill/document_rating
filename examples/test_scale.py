__author__ = 'NLP-PC'
from sklearn import preprocessing
import numpy as np

X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3])
X_normalized = preprocessing.normalize(X_train, norm='l2')
# X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_normalized)
print('OK')
