__author__ = 'NLP-PC'
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


def draw_line(x, y, x_labels, y_labels, title):
    plt.plot(x, y, 'o-')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.show()


def draw_linear_regression(X_test, Y_test, predict):
    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, predict, color='blue', linewidth=3)
    plt.show()


def show_common_term(corpus, lexicon, threshold=1):
    for sent_list in corpus:
        word_count = [sent_list.count(w[0]) for w in lexicon]
        word_count = np.array(word_count)
        occur_times = (word_count >= 1).sum()
        print(' '.join(sent_list), np.array(lexicon)[word_count >= 1, 0], occur_times)


if __name__ == "__main__":
    v = [0.688022284123, 0.740947075209, 0.74930362117, 0.757660167131, 0.757660167131, 0.771587743733, 0.782729805014,
         0.779944289694, 0.782729805014, 0.791086350975, 0.793871866295, 0.782729805014, 0.788300835655]
    x = np.linspace(100, 1300, len(v))
    draw_line(x, v, 'Feature number', 'Accuracy', 'Number of People')

    X_test, Y_test, predict = [1, 2, 3], [1, 4, 9], [1, 3.5, 8.6]
    draw_linear_regression(X_test, Y_test, predict)
