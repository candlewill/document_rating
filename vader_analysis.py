__author__ = 'nobody'
from load_data import load_pickle
from file_name import get_file_path

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_movie_news_articles4.p')

from evaluate import evaluate

print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
