__author__ = 'hs'
from load_data import load_pickle
from preprocess_imdb import clean_str
from word2vec_fn import make_idx_data
# Test
[idx_data, ratings] = load_pickle('./data/corpus/vader/vader_processed_data_tweets.p')
print(idx_data[2])
print(ratings[2])

W = load_pickle('./data/corpus/vader/embedding_matrix_tweets.p')
print(len(W[1]))

request_text = 'I want to go to school and you what about you ?'
request_text = clean_str(request_text)
print(request_text)
word_idx_map = load_pickle('./data/corpus/vader/word_idx_map_movie_reviews.p')
idx_request_text = make_idx_data(request_text, word_idx_map)
