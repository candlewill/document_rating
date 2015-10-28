__author__ = 'NLP-PC'
from load_data import load_pickle
from save_data import dump_picle


def convert(source_file):
    s = load_pickle(source_file)
    dump_picle(s, str(source_file)[:-2] + '_v2.7.p', protocol=2)


convert('./web_api/embedding_matrix_CVAT.p')
convert('./web_api/word_idx_map_CVAT.p')
