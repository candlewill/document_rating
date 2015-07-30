__author__ = 'NLP-PC'
import numpy as np
from sklearn import cross_validation
from fusion_model import linear_fusion, linear_fusion_sqr, nonlinear_max_fusion
from load_data import load_corpus, load_lexicon, load_mark
from file_name import get_file_path
from regression import linear_regression

def cv(data, target):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data, target, test_size=0.2, random_state=0)
    linear_regression(X_train, X_test, Y_train, Y_test, plot=False)


corpus = load_corpus(get_file_path('cn_corpus'))
lexicon = load_lexicon(get_file_path('lexicon'))
mark = load_mark(get_file_path('mark'))

# valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion(corpus, lexicon, mark)
# valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_sqr(corpus, lexicon, mark)
valence_mean, valence_true, arousal_mean, arousal_true = nonlinear_max_fusion(corpus, lexicon, mark)

cv(valence_mean, valence_true)
cv(arousal_mean, arousal_true)
print('OK')