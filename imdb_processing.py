__author__ = 'NLP-PC'
import os
from load_data import load_embeddings, load_pickle
from file_name import get_file_path
import codecs
import string
from word2vec_fn import build_sentence_matrix
from save_data import dump_picle
from preprocess_imdb import clean_str
from word2vec_fn import make_idx_data
from word2vec_fn import build_embedding_matrix
import numpy as np
from collections import defaultdict


# Process Flow: make vocab -> make word index map -> make indexed data and W -> model -> result


def load_data(file_dir):
    file_names = os.listdir(file_dir)
    data = []
    # Do some very minor text preprocessing
    def cleanText(text):
        for c in string.punctuation:
            corpus = text.lower().replace('\n', '').replace('.', ' ').replace(',', ' ').replace(c, '').split()
        return corpus

    length = len(file_names)
    for file_name in file_names:
        text = ' '.join(codecs.open(os.path.join(file_dir, file_name), 'r', 'utf-8').readlines())
        data.append(cleanText(text))
    return data, length


def get_vocab(file_dir):
    vocab = defaultdict(float)

    def load_data(file_dir):
        file_names = os.listdir(file_dir)
        for file_name in file_names:
            text = ' '.join(codecs.open(os.path.join(file_dir, file_name), 'r', 'utf-8').readlines())
            for word in clean_str(text).split():
                vocab[word] += 1

    load_data(file_dir + 'pos/')
    load_data(file_dir + 'neg/')
    return vocab


def prepare_data(file_dir, word_idx_map):
    def load_data(file_dir):
        file_names = os.listdir(file_dir)
        data = []
        length = len(file_names)
        for file_name in file_names:
            text = ' '.join(codecs.open(os.path.join(file_dir, file_name), 'r', 'utf-8').readlines())
            data.append(clean_str(text))
        idx_data = make_idx_data(data, word_idx_map, max_len=200, kernel_size=5)
        return idx_data, length

    pos_idx_data, pos_length = load_data(file_dir + 'pos/')
    print(pos_idx_data.shape, pos_length)
    neg_idx_data, neg_length = load_data(file_dir + 'neg/')
    print(neg_idx_data.shape, neg_length)
    data = np.concatenate((pos_idx_data, neg_idx_data), axis=0)
    print(data.shape)
    return data, pos_length, neg_length


if __name__ == '__main__':
    ########################################## config ########################################
    file_dir = 'E:/研究/Data/IMDB/aclImdb/train/' if os.name == 'nt' else '/home/hs/Data/imdb/aclImdb/train/'
    vec_dim = 300
    ##########################################################################################

    # get vocab and save to pickle
    vocab = get_vocab(file_dir)
    dump_picle(vocab, './data/tmp/vocab.p')
    print('OK')
    a = load_pickle('./data/tmp/vocab.p')
    for i in a:
        print(i)
    print(len(a))
    exit()
    # end

    # make word index map
    vocab = load_pickle('./data/tmp/vocab.p')
    W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), vocab, k=300)
    dump_picle(word_idx_map, get_file_path('word_idx_map'))
    print('dump word_idx_map successful')
    dump_picle(W, '/home/hs/Data/embedding_matrix.p')
    print('OK')
    exit()
    # make word index map end

    word_idx_map = load_pickle(get_file_path('word_idx_map'))
    print(len(word_idx_map))
    for i in word_idx_map:
        print(i)
    exit()

    word_idx_map = load_pickle(get_file_path('word_idx_map'))
    data, pos_length, neg_length = prepare_data(file_dir, word_idx_map)
    dump_picle([data, pos_length, neg_length], get_file_path('imdb_processed_data'))
    exit()

    # model = load_embeddings('google_news')
    # print('Loading word2vec complete')
    #
    # pos_data, pos_length = load_data(file_dir + 'pos')
    # pos_sentences_matrix = build_sentence_matrix(model, pos_data, maxlen=200, dim=vec_dim)
    #
    # print('compute pos data ok')
    #
    # dump_picle((pos_sentences_matrix, pos_length), file_dir + 'pos.p', protocol=4)
    # print('save pos data ok')
    #
    # neg_data, neg_length = load_data(file_dir + 'neg')
    # neg_sentences_matrix = build_sentence_matrix(model, neg_data, maxlen=200, dim=vec_dim)
    # print('compute neg data ok')
    #
    # dump_picle((neg_sentences_matrix, neg_length), file_dir + 'neg.p', protocol=4)
    # print('save neg data ok')
