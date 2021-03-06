__author__ = 'NLP-PC'
import random

import numpy as np

from file_name import get_file_path
from load_data import load_embeddings
from word2vec_fn import buill_word_vector
from cross_validation import cv
from load_data import load_vader
from affective_score_vader import screen_data
from load_data import load_anew

print('start')
model = load_embeddings('google_news')

corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(np.histogram(ratings, bins=range(10)))
print(len(model.vocab.keys()))
vecs = np.concatenate([buill_word_vector(text, model, size=300) for text in corpus])
print(vecs[1])
cv(vecs, ratings, multivariant=True)

vecs = None
ratings = None
corpus, ratings = load_vader(['tweets'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(np.histogram(ratings, bins=range(10)))
print(len(model.vocab.keys()))
vecs = np.concatenate([buill_word_vector(text, model, size=300) for text in corpus])
print(vecs[1])
cv(vecs, ratings, multivariant=True)

vecs = None
ratings = None
corpus, ratings = load_vader(['movie_reviews'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(np.histogram(ratings, bins=range(10)))
print(len(model.vocab.keys()))
vecs = np.concatenate([buill_word_vector(text, model, size=300) for text in corpus])
print(vecs[1])
cv(vecs, ratings, multivariant=True)

vecs = None
ratings = None
corpus, ratings = load_vader(['product_reviews'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(np.histogram(ratings, bins=range(10)))
print(len(model.vocab.keys()))
vecs = np.concatenate([buill_word_vector(text, model, size=300) for text in corpus])
print(vecs[1])
cv(vecs, ratings, multivariant=True)

vecs = None
ratings = None
corpus, ratings = load_vader(['news_articles'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
print(np.histogram(ratings, bins=range(10)))
print(len(model.vocab.keys()))
vecs = np.concatenate([buill_word_vector(text, model, size=300) for text in corpus])
print(vecs[1])
cv(vecs, ratings, multivariant=True)
exit()
# from save_data import dump_picle
# dump_picle(model.key(), get_file_path('words_in_wordvec'))
# print('ok')
#
# # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
# # print(model.doesnt_match("breakfast cereal dinner lunch".split()))
# # print(model.similarity('woman', 'man'))
# # print(model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'], topn=10))
# # print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
#
# from load_data import load_pickle
# words = load_pickle(get_file_path('words_in_wordvec'))
# print(words)

class Sentence(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def toarray(self):
        return self.corpus

    def rand(self):
        random.shuffle(self.corpus)
        return self.corpus


################################################
corpus, ratings = load_vader(['tweets', 'movie_reviews', 'product_reviews', 'news_articles'])
# corpus, ratings = load_vader(['news_articles'])
lexicon_name = get_file_path('anew')
words, valences, _ = load_anew(lexicon_name)
corpus, ratings = screen_data(corpus, ratings, words)
ratings = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5

# Train word2vec
from word2vec_fn import train_wordvecs

sentence = Sentence(corpus)
train_wordvecs(sentence, './data/vader_wordvecs.w2v')
#################################################
