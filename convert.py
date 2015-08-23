__author__ = 'NLP-PC'
from load_data import load_pickle
from save_data import dump_picle


def convert(source_file):
    s = load_pickle(source_file)
    dump_picle(s, str(source_file)[:-2] + '_v2.7.p', protocol=2)


convert('./data/tmp/CVAT_processed_data.p')
convert('./data/tmp/embedding_matrix_CVAT.p')
