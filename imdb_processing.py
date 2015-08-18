__author__ = 'NLP-PC'
import os
from load_data import load_embeddings
from file_name import get_file_path
import codecs
import string
from word2vec_fn import build_sentence_matrix
from save_data import dump_picle
from preprocess_imdb import clean_str
from word2vec_fn import make_idx_data
from word2vec_fn import build_embedding_matrix

########################################## config ########################################
file_dir = 'E:/研究/Data/IMDB/aclImdb/train/' if os.name == 'nt' else '/home/hs/Data/imdb/aclImdb/train/'
vec_dim = 300
##########################################################################################

# make word index map
W, word_idx_map = build_embedding_matrix(load_embeddings('google_news'), k=300)
dump_picle(word_idx_map, get_file_path('word_idx_map'))

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


def prepare_data(word_vecs, file_dir):
    def load_data(file_dir):
        file_names = os.listdir(file_dir)
        data = []
        length = len(file_names)
        for file_name in file_names:
            text = ' '.join(codecs.open(os.path.join(file_dir, file_name), 'r', 'utf-8').readlines())
            data.append(clean_str(text))
        idx_data = make_idx_data(data, word_idx_map, max_len=200, kernel_size=5)
        return idx_data, length

    load_data(file_dir + 'pos/')


prepare_data(None, file_dir)
exit()

model = load_embeddings('google_news')
print('Loading word2vec complete')


pos_data, pos_length = load_data(file_dir + 'pos')
pos_sentences_matrix = build_sentence_matrix(model, pos_data, maxlen=200, dim=vec_dim)

print('compute pos data ok')

dump_picle((pos_sentences_matrix, pos_length), file_dir + 'pos.p', protocol=4)
print('save pos data ok')

neg_data, neg_length = load_data(file_dir + 'neg')
neg_sentences_matrix = build_sentence_matrix(model, neg_data, maxlen=200, dim=vec_dim)
print('compute neg data ok')

dump_picle((neg_sentences_matrix, neg_length), file_dir + 'neg.p', protocol=4)
print('save neg data ok')
