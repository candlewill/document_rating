__author__ = 'NLP-PC'
import os


def get_file_path(filename=None):
    out = None
    os_name = os.name
    if filename == 'cn_corpus':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'corpus_raw')
    elif filename == 'mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'mark.csv')
    elif filename == 'lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'lexicon.txt')
    elif filename == 'neural_cand':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'expand', 'neural_cand.txt')
    elif filename == 'log':
        out = os.path.join('.', 'log', 'logs.log')
    elif filename == 'anew':
        out = os.path.join('.', 'data', 'corpus', 'anew_seed.txt')
    elif filename == 'normalized_lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'normalized_lexicon.txt')
    elif filename == 'normalized_mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'normalized_mark.csv')
    elif filename == 'normalized_onezero_lexicon':
        out = os.path.join('.', 'data', 'corpus', 'cn_lexicon', 'normalized_onezero_lexicon.txt')
    elif filename == 'normalized_onezero_mark':
        out = os.path.join('.', 'data', 'corpus', 'cn', 'normalized_onezero_mark.csv')
    elif filename == 'test_doc2vec':
        if os_name == 'posix':  # ubuntu
            out = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec')
        elif os_name == 'nt':  # windows
            out = os.path.join('D:\\', 'chinese_word2vec', 'test_doc2vec')
    elif filename == 'test_doc2vec_model':
        if os_name == 'posix':
            out = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec', 'imdb.d2v')
        elif os_name == 'nt':
            out = os.path.join('D:\\', 'chinese_word2vec', 'test_doc2vec', 'docvecs', 'imdb.d2v')
    elif filename == 'cn_word2vec':
        posix = os.path.join('/', 'home', 'hs', 'Data', 'test_doc2vec', 'cn_word2vec', 'wiki.zh.fan.vector')
        nt = os.path.join('D:\\', 'chinese_word2vec', 'wiki.zh.fan.vector')
        if os_name == 'posix':
            out = posix
        elif os_name == 'nt':
            out = nt
    elif filename == 'words_in_wordvec':
        out = os.path.join('.', 'data', 'tmp', 'words_in_wordvec.p')
    elif filename == 'wordvecs_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'wordvecs_CVAT.w2v')
    elif filename == 'docvecs_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'docvecs_CVAT.d2v')
    elif filename == 'NN_input_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'NN_input_CVAT.p')
    elif filename == 'NN_output_CVAT':
        out = os.path.join('.', 'data', 'tmp', 'NN_output_CVAT.p')
    elif filename == 'CVAT_sentence_matrix_400':
        posix = os.path.join('/', 'home', 'hs', 'Data', 'CVAT_sentence_matrix_400.p')
        nt = 'D:/chinese_word2vec/CVAT_sentence_matrix_400.p'
        if os_name == 'posix':
            out = posix
        elif os_name == 'nt':
            out = nt
    elif filename == 'google_news':
        posix = '/home/hs/Data/Word_Embeddings/google_news.bin'
        nt = 'D:/Word_Embeddings/GoogleNews-vectors-negative300.bin'
        if os_name == 'posix':
            out = posix
        elif os_name == 'nt':
            out = nt
    elif filename == 'word_idx_map':
        out = './data/tmp/word_idx_map.p'
    elif filename == 'imdb_processed_data':
        out = './data/tmp/imdb_processed_data.p'
    elif filename == 'CVAT_Vocab':
        out = './data/tmp/CVAT_Vocab.p'
    elif filename == 'word_idx_map_CVAT':
        out = './data/tmp/word_idx_map_CVAT.p'
    elif filename == 'CVAT_processed_data':
        out = './data/tmp/CVAT_processed_data.p'
    else:
        raise Exception('Wrong filename')
    return out
