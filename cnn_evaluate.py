__author__ = 'NLP-PC'

from load_data import load_pickle
from file_name import get_file_path

(Y_test, predict) = load_pickle('./data/corpus/vader/cnn_tweets_result.p')

from evaluate import evaluate

print(Y_test)
print(predict)
evaluate(Y_test, predict, 'tweets')
