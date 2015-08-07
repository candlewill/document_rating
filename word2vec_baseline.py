__author__ = 'NLP-PC'
import gensim
import os
import time
from file_name import get_file_path

model_file = get_file_path('cn_word2vec')
model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False)
print('load model complete')

print(model['电脑'])

# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))
# print(model.similarity('woman', 'man'))
# print(model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'], topn=10))
# print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
