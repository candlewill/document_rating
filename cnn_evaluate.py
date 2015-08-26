__author__ = 'NLP-PC'

from load_data import load_pickle
from file_name import get_file_path
from evaluate import evaluate

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles.p')
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'news_articles')

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles1.p')
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'news_articles')

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles2.p')
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'news_articles')

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles3.p')
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'news_articles')

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles4.p')
print(Y_test)
print(predict)
evaluate(Y_test, predict, 'news_articles')
