__author__ = 'NLP-PC'
import gensim
import os
import time
from file_name import get_file_path
from load_data import load_corpus, load_lexicon, load_mark
from load_data import load_embeddings
from word2vec_fn import buill_word_vector
from word2vec_fn import gold_valence_arousal
import numpy as np
from sklearn import cross_validation
from cross_validation import cv

model = load_embeddings('zh_tw')

print(len(model.vocab.keys()))
corpus = load_corpus(get_file_path('cn_corpus'))
mark = load_mark(get_file_path('mark'))
vecs = np.concatenate([buill_word_vector(text, model) for text in corpus])
valence, arousal = gold_valence_arousal(corpus, mark)
cv(vecs, valence, multivariant=True)
cv(vecs, arousal, multivariant=True)
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

################################################
# Train word2vec
# from word2vec_fn import Sentence, train_wordvecs
# sentence = Sentence(get_file_path('cn_corpus'))
# train_wordvecs(sentence)
#################################################
