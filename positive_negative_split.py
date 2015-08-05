__author__ = 'NLP-PC'


def get_pos_neg_va(corpus, lexicon, mark):
    pos_valence_mean = []
    neg_valence_mean = []
    valence_true = []
    pos_arousal_mean = []
    neg_arousal_mean = []
    arousal_true = []

    def VA_mean(text):
        pos_sum_valence = 1
        neg_sum_valence = 1
        pos_sum_arousal = 1
        neg_sum_arousal = 1
        pos_count_v = 0
        neg_count_v = 0
        pos_count_a = 0
        neg_count_a = 0
        avg = 5
        for word in text:
            for l in lexicon:
                if word == l[0]:
                    # if l[1] > 9:
                    #     l[1] = 9
                    # if l[1] < 1:
                    #     l[1] = 1
                    # if l[2] > 9:
                    #     l[2] = 9
                    # if l[2] < 1:
                    #     l[2] = 1
                    if l[1] >= avg:
                        pos_count_v = pos_count_v + 1
                        pos_sum_valence = pos_sum_valence * l[1]
                    else:
                        neg_count_v = neg_count_v + 1
                        neg_sum_valence = neg_sum_valence * l[1]
                    if l[2] >= avg:
                        pos_count_a = pos_count_a + 1
                        pos_sum_arousal = pos_sum_arousal * l[1]
                    else:
                        neg_count_a = neg_count_a + 1
                        neg_sum_arousal = neg_sum_arousal * l[1]
        pos_valence = (pos_sum_valence ** (1. / pos_count_v) if pos_count_v != 0 else avg)
        neg_valence = (neg_sum_valence ** (1. / neg_count_v) if neg_count_v != 0 else avg)
        pos_arousal = (pos_sum_arousal ** (1. / pos_count_a) if pos_count_a != 0 else avg)
        neg_arousal = (neg_sum_arousal ** (1. / neg_count_a) if neg_count_a != 0 else avg)
        return [pos_valence, neg_valence, pos_arousal, neg_arousal]

    for (i, text) in enumerate(corpus):
        pos_valence, neg_valence, pos_arousal, neg_arousal = VA_mean(text)
        pos_valence_mean.append(pos_valence)
        neg_valence_mean.append(neg_valence)
        pos_arousal_mean.append(pos_arousal)
        neg_arousal_mean.append(neg_arousal)
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))

        valence_true.append(mark[ind][1])
        arousal_true.append(mark[ind][2])
    return pos_valence_mean, neg_valence_mean, valence_true, pos_arousal_mean, neg_arousal_mean, arousal_true
