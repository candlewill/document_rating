__author__ = 'NLP-PC'

from load_data import load_pickle
from file_name import get_file_path

(Y_test, predict) = load_pickle(get_file_path('NN_output_CVAT'))

from evaluate import evaluate

print(Y_test)
print(predict)
evaluate(Y_test, predict, 'Result of CNN')
