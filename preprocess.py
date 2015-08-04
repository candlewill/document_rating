__author__ = 'NLP-PC'
import numpy as np


def scaling(num_list):
    # Note: the type of the parameter is np.array
    # Function: To normalize data
    result = []
    for num in num_list:
        result.append((num - 5) / 5)
    return result


def scaling_onezero(num_list):
    # Note: the type of the parameter is np.array
    # Function: To normalize data
    result = []
    for num in num_list:
        result.append(num / 9)
    return result


if __name__ == '__main__':
    from load_data import load_lexicon
    from load_data import load_mark
    from file_name import get_file_path
    from save_data import save_csv

    lexicon = load_lexicon(get_file_path('lexicon'))
    mark = load_mark(get_file_path('mark'))
    lexicon = np.array(lexicon)
    mark = np.array(mark)
    #####################################
    lexicon[:, 1] = scaling_onezero(np.array(lexicon[:, 1], dtype=float))
    lexicon[:, 2] = scaling_onezero(np.array(lexicon[:, 2], dtype=float))
    mark[:, 1] = scaling_onezero(np.array(mark[:, 1], dtype=float))
    mark[:, 2] = scaling_onezero(np.array(mark[:, 2], dtype=float))
    ######################################
    save_csv(lexicon, get_file_path('normalized_onezero_lexicon'))
    save_csv(mark, get_file_path('normalized_onezero_mark'))
