__author__ = 'NLP-PC'
# Dataset	dataset size	vocabulary size	average sentence length	ratings
from load_data import load_vader, load_corpus
from file_name import get_file_path
import numpy as np


def statistic(texts):
    avg_length, vocab = 0, set()
    length_list = []
    for text in texts:
        if type(text) is not list:
            text = text.split()
        length_list.append(len(text))
        vocab = vocab.union(set(text))
        # if len(text)>200:
        #     print(text)
    avg_length = np.average(np.array(length_list))
    return avg_length, len(vocab)


if __name__ == '__main__':
    # (['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
    tweets, _ = load_vader(['tweets'])
    movie, _ = load_vader(['movie_reviews'])
    amazon, _ = load_vader(['product_reviews'])
    NYT, _ = load_vader(['news_articles'])
    cvat = load_corpus(get_file_path('cn_corpus'))

    print(statistic(tweets))
    print(statistic(movie))
    print(statistic(amazon))
    print(statistic(NYT))
    print(statistic(cvat))
