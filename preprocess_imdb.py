__author__ = 'NLP-PC'
from string import punctuation


def clean_str(sentence):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    for p in list(punctuation):
        sentence = sentence.replace(p, '')
    return sentence.strip().lower()


if __name__ == '__main__':
    print(clean_str('we ok and you? AFE and Who a'))
