__author__ = 'NLP-PC'
import numpy as np
from sklearn import cross_validation
from fusion_model import linear_fusion, linear_fusion_sqr, nonlinear_max_fusion, linear_fusion_tf, linear_fusion_tfidf, \
    linear_fusion_geo, linear_fusion_geo_tf, linear_fusion_geo_tfidf
from load_data import load_corpus, load_lexicon, load_mark
from file_name import get_file_path
from regression import linear_regression, linear_regression_multivariant
from positive_negative_split import get_pos_neg_va
from load_data import combine_lexicon

def cv(data, target, multivariant=False):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data, target, test_size=0.2, random_state=0)
    if multivariant is False:
        linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
    else:
        linear_regression_multivariant(X_train, X_test, Y_train, Y_test, cost_fun='Ridge_Regression')


normalize = True
corpus = load_corpus(get_file_path('cn_corpus'))
# lexicon = load_lexicon(get_file_path('lexicon'))
mark = load_mark(get_file_path('mark'))
lexicon = combine_lexicon(get_file_path('lexicon'), get_file_path('neural_cand'))

# # the following could use to check the same words in corpus and lexicon
# from visualization import show_common_term
# show_common_term(corpus, lexicon)
# exit()

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_sqr(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = nonlinear_max_fusion(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_tf(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_tfidf(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_geo(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_geo_tf(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

valence_mean, valence_true, arousal_mean, arousal_true = linear_fusion_geo_tfidf(corpus, lexicon, mark)
print('start.....')
cv(valence_mean, valence_true, multivariant=False)
cv(arousal_mean, arousal_true, multivariant=False)
print('OK')

# Note: change the avg, when changing using normalized data or not
pos_valence_mean, neg_valence_mean, valence_true, pos_arousal_mean, neg_arousal_mean, arousal_true = get_pos_neg_va(
    corpus, lexicon, mark)
valence_mean = np.array([pos_valence_mean, neg_valence_mean]).T
arousal_mean = np.array([pos_arousal_mean, neg_arousal_mean]).T

print('start.....')
cv(valence_mean, valence_true, multivariant=True)
cv(arousal_mean, arousal_true, multivariant=True)
print('OK')
