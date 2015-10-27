__author__ = 'hs'
from load_data import load_pickle

# Test
[idx_data, ratings] = load_pickle('./data/corpus/vader/vader_processed_data_tweets.p')
print(idx_data[2])
print(ratings[2])

W = load_pickle('./data/corpus/vader/embedding_matrix_tweets.p')
print(len(W[1]))