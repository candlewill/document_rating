__author__ = 'NLP-PC'

# mean of valence and arousal
def linear_fusion(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []
    arousal_mean = []
    arousal_true = []

    def VA_mean(text):
        sum_valence = 0
        sum_arousal = 0
        count = 0
        for word in text:
            for l in lexicon:
                if word == l[0]:
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1
                    count = count + 1
                    sum_valence = sum_valence + l[1]
                    sum_arousal = sum_arousal + l[2]
        return [5., 5.] if count == 0 else [sum_valence / count, sum_arousal / count]

    for (i, text) in enumerate(corpus):
        V, A = VA_mean(text)
        valence_mean.append(V)
        arousal_mean.append(A)
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))

        valence_true.append(mark[ind][1])
        arousal_true.append(mark[ind][2])
    return valence_mean, valence_true, arousal_mean, arousal_true

# sum of sqr, just like the formula in the paper of Malandrakis except the sign()
def linear_fusion_sqr(corpus, lexicon, mark):
    valence_mean = []
    valence_true = []
    arousal_mean = []
    arousal_true = []

    def VA_sqr_mean(text):
        sum_valence = 0
        sum_arousal = 0
        sum_valence_sqr = 0
        sum_arousal_sqr = 0
        for word in text:
            for l in lexicon:
                if word == l[0]:
                    if l[1] > 9:
                        l[1] = 9
                    if l[1] < 1:
                        l[1] = 1
                    if l[2] > 9:
                        l[2] = 9
                    if l[2] < 1:
                        l[2] = 1
                    sum_valence_sqr = sum_valence_sqr + l[1]**2
                    sum_arousal_sqr = sum_arousal_sqr + l[2]**2
                    sum_valence = sum_valence + l[1]
                    sum_arousal = sum_arousal + l[2]
        return [5., 5.] if sum_valence == 0 else [sum_valence_sqr / sum_valence, sum_arousal_sqr / sum_arousal]

    for (i, text) in enumerate(corpus):
        V, A = VA_sqr_mean(text)
        valence_mean.append(V)
        arousal_mean.append(A)
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))

        valence_true.append(mark[ind][1])
        arousal_true.append(mark[ind][2])
    return valence_mean, valence_true, arousal_mean, arousal_true

def nonlinear_max_fusion(corpus, lexicon, mark):
    valence_max = []
    valence_true = []
    arousal_max = []
    arousal_true = []

    def VA_max(text):
        max_valence = 0
        max_arousal = 0
        for word in text:
            for l in lexicon:
                if word == l[0]:
                    if l[1] > max_valence:
                        max_valence = l[1]
                    if l[2] > max_arousal:
                        max_arousal = l[2]
        return [5., 5.] if max_valence == 0 else [max_valence, max_arousal]

    for (i, text) in enumerate(corpus):
        V, A = VA_max(text)
        valence_max.append(V)
        arousal_max.append(A)
        try:
            ind = [item[0] for item in mark].index(i + 1)
        except ValueError:
            raise Exception('File not found. NO. %i' % (i + 1))

        valence_true.append(mark[ind][1])
        arousal_true.append(mark[ind][2])
    return valence_max, valence_true, arousal_max, arousal_true