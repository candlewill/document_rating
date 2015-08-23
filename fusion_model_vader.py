__author__ = 'NLP-PC'


def linear_fusion(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    count = count + 1
                    sum_valence = sum_valence + lexicon[line]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_mean.append(V)
        valence_true.append(mark[i])
    return valence_mean, valence_true


from load_data import load_pickle

idfs = load_pickle('./data/vocab_idf.p')


def tfidf(t, d):
    d = d.split()
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    # idf = sp.log(float(len(D)) / (len([doc.split() for doc in D if t in doc.split()])))
    return tf * idfs[t]


def tf(t, d):
    d = d.split()
    tf = float(d.count(t)) / float(len(d))
    return tf


def linear_fusion_sqr(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        sum_valence_sqr = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    count = count + 1
                    sum_valence_sqr = sum_valence_sqr + lexicon[line] ** 2
                    sum_valence = sum_valence + lexicon[line]
        return 5 if count == 0 else sum_valence_sqr / sum_valence

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_mean.append(V)
        valence_true.append(mark[i])
    return valence_mean, valence_true


def nonlinear_max_fusion(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []

    def VA_mean(text):
        max_valence = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    if lexicon[word] > max_valence:
                        max_valence = lexicon[word]
        return 5 if max_valence == 0 else max_valence

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_mean.append(V)
        valence_true.append(mark[i])
    return valence_mean, valence_true


def linear_fusion_tf(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tf = tf(word, corpus[i])
                    count = count + word_tf
                    sum_valence = sum_valence + word_tf * lexicon[word]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_mean.append(V)
        valence_true.append(mark[i])
    return valence_mean, valence_true


def linear_fusion_tfidf(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tfidf = tfidf(word, corpus[i])
                    count = count + word_tfidf
                    sum_valence = sum_valence + word_tfidf * lexicon[word]
        return 5 if count == 0 else sum_valence / count

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    return valence_pred, valence_true


def linear_fusion_geo(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    count = count + 1
                    sum_valence = sum_valence * lexicon[line]
        return 5 if count == 0 else sum_valence ** (1. / count)

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    return valence_pred, valence_true


def linear_fusion_geo_tf(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    word_tf = tf(word, corpus[i])
                    count = count + word_tf
                    sum_valence = sum_valence * (lexicon[word] ** word_tf)
        out = 5 if count == 0 else sum_valence ** (1. / count)
        print(out)
        return out

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    return valence_pred, valence_true


def linear_fusion_geo_tfidf(corpus, lexicon, mark):
    valence_pred = []
    valence_true = []

    def VA_mean(text):
        sum_valence = 1.
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                word_tfidf = tfidf(word, corpus[i])
                if word == line:
                    count = count + word_tfidf
                    sum_valence = sum_valence * (lexicon[word] ** word_tfidf)
        out = 5 if count == 0 else sum_valence ** (1. / count)
        print(out)
        return out

    for i, text in enumerate(corpus):
        V = VA_mean(text)
        valence_pred.append(V)
        valence_true.append(mark[i])
    print(valence_true[:200])
    print(valence_pred[:200])
    return valence_pred, valence_true


from preprocess_imdb import clean_str


def process(corpus):
    return [clean_str(sent) for sent in corpus]


from regression import linear_regression, linear_regression_multivariant
from sklearn import cross_validation


def cv(data, target, multivariant=False):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data, target, test_size=0.2, random_state=0)
    if multivariant is False:
        linear_regression(X_train, X_test, Y_train, Y_test, plot=False)
    else:
        linear_regression_multivariant(X_train, X_test, Y_train, Y_test, cost_fun='Ridge_Regression')


if __name__ == '__main__':

    from load_data import load_vader

    normalize = True
    corpus, ratings = load_vader(['movie_reviews'])
    corpus = process(corpus)
    # lexicon = load_lexicon(get_file_path('lexicon'))
    from load_data import load_anew
    from file_name import get_file_path
    import numpy as np

    words, valences, _ = load_anew(get_file_path('anew'))
    mark = np.array(ratings) + np.ones(len(ratings), dtype=float) * 5
    lexicon = dict()
    for i, word in enumerate(words):
        lexicon[word] = valences[i]

    # # the following could use to check the same words in corpus and lexicon
    # from visualization import show_common_term
    # show_common_term(corpus, lexicon)
    # exit()

    # valence_mean, valence_true = linear_fusion(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')


    # valence_mean, valence_true = linear_fusion_sqr(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')


    # valence_mean, valence_true = nonlinear_max_fusion(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')

    # valence_mean, valence_true = linear_fusion_tf(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')
    #
    # valence_mean, valence_true = linear_fusion_tfidf(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')
    #
    # valence_mean, valence_true = linear_fusion_geo(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')

    ########################

    # valence_mean, valence_true = linear_fusion_geo_tf(corpus, lexicon, mark)
    # print('start.....')
    # cv(valence_mean, valence_true, multivariant=False)
    # print('OK')

    valence_mean, valence_true = linear_fusion_geo_tfidf(corpus, lexicon, mark)
    print('start.....')
    cv(valence_mean, valence_true, multivariant=False)
    print('OK')
