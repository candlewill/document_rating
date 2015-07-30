__author__ = 'NLP-PC'
import os

def get_file_path(filename=None):
    out = None
    if filename == 'cn_corpus':
        out = os.path.join('.','data','corpus','cn','corpus_raw')
    elif filename == 'mark':
        out = os.path.join('.','data','corpus','cn','mark.csv')
    elif filename == 'lexicon':
        out = os.path.join('.','data','corpus','cn_lexicon','lexicon.txt')
    elif filename == 'neural_cand':
        out = os.path.join('.','data','corpus','cn_lexicon','expand','neural_cand.txt')
    elif filename == 'log':
        out = os.path.join('.','log','logs.log')
    elif filename == 'anew':
        out = os.path.join('.','data', 'corpus', 'anew_seed.txt')
    else:
        raise Exception('Wrong filename')
    return out