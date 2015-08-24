__author__ = 'NLP-PC'
from load_data import load_pickle
from evaluate import evaluate
import random
from regression import linear_regression, linear_regression_multivariant
from sklearn import cross_validation

mean_ratings, tf_means, tfidf_means, geos, tf_geos, tfidf_geos, ratings = load_pickle('./data/vader_out.p')

# size = 720
# slice_idx = random.sample(range(len(ratings)), size)  # 从list中随机获取size个元素，作为一个片断返回
# mean_ratings, tf_means, tfidf_means, geos, tf_geos, tfidf_geos, ratings = mean_ratings[slice_idx], tf_means[slice_idx], \
#                                                                           tfidf_means[
#                                                                               slice_idx], geos[slice_idx], tf_geos[
#                                                                               slice_idx], tfidf_geos[slice_idx], \
#                                                                           ratings[slice_idx]


evaluate(ratings, mean_ratings, 'mean_ratings')
evaluate(ratings, tf_means, 'tf_means')
evaluate(ratings, tfidf_means, 'tfidf_means')
evaluate(ratings, geos, 'geos')
evaluate(ratings, tf_geos, 'tf_geos')
evaluate(ratings, tfidf_geos, 'tfidf_geos')

################################################ Regression Methods ##########################################
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(mean_ratings, ratings, test_size=0.2,
#                                                                      random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(tf_means, ratings, test_size=0.2, random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(tfidf_means, ratings, test_size=0.2,
#                                                                      random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(geos, ratings, test_size=0.2, random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(tf_geos, ratings, test_size=0.2, random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(tfidf_geos, ratings, test_size=0.2, random_state=0)
# linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
