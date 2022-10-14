import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download("brown")
from nltk.corpus import brown

import string
import pickle
import math
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD


def preprocess():
    translator = str.maketrans("", "", string.punctuation)
    stop_words = set(stopwords.words("english"))
    clean_text = []
    for word in brown.words():
        word = word.lower()
        if (
            word not in stop_words and word.translate(translator) != ""
        ):  # remove stopwords and punctuation
            clean_text.append(word)
    return clean_text


def generate_vocab():
    text = preprocess()
    fDist = nltk.FreqDist(text)
    fDist_5000 = fDist.most_common(5000)
    print("the 5 most common words: \n", fDist_5000[:5])
    print("the 5 least common words: \n", fDist_5000[-5:])

    with open("table1.plk", "rb") as f:
        table1 = set(pickle.load(f))

    fDist_5000_w = set(list(zip(*fDist_5000))[0])

    # find the intersection between W and table1 - P and S
    intersect = fDist_5000_w.intersection(table1)
    with open("table1_sim.plk", "rb") as f:
        table1_sim = pickle.load(f)

    i = 0
    s = []
    p = {}
    for key in table1_sim.keys():
        v, w = key
        if v in intersect and w in intersect:
            p[(v, w)] = i
            s.append(table1_sim[key])
            i += 1

    vocab = list(fDist_5000_w.union(table1))
    print(f"|W| = {len(vocab)}.\n")

    return vocab, p, np.array(s)


def calc_sim(m, v, w, vocab_index):
    vec_v = m[vocab_index[v], :]
    vec_w = m[vocab_index[w], :]
    return np.matmul(vec_v, vec_w) / (norm(vec_v, 2) * norm(vec_w, 2))


if __name__ == "__main__":
    vocab, p, s = generate_vocab()
    vocab_index = {vocab[i]: i for i in range(len(vocab))}

    # word-context matrix
    corpus = brown.words()
    m1 = np.zeros((len(vocab), len(vocab)))
    for i in range(len(corpus) - 1):
        cur_w = corpus[i].lower()
        next_w = corpus[i + 1].lower()
        if cur_w in vocab_index and next_w in vocab_index:
            m1[vocab_index[cur_w], vocab_index[next_w]] += 1
            if next_w != cur_w:
                m1[vocab_index[next_w], vocab_index[cur_w]] += 1

    # PPMI
    m1_plus = np.zeros_like(m1)
    total = np.sum(m1)
    marginal = np.sum(m1, axis=0)
    for v in vocab_index.keys():
        for w in vocab_index.keys():
            if m1[vocab_index[v], vocab_index[w]] == 0:
                continue
            p_v_w = m1[vocab_index[v], vocab_index[w]] / total
            p_v = marginal[vocab_index[v]] / total
            p_w = marginal[vocab_index[w]] / total
            m1_plus[vocab_index[v], vocab_index[w]] = max(math.log2(p_v_w / (p_v * p_w)), 0)

    # Latent Semantic Analysis
    svd_10 = TruncatedSVD(n_components=10)
    m2_10 = svd_10.fit_transform(m1_plus)

    svd_100 = TruncatedSVD(n_components=100)
    m2_100 = svd_100.fit_transform(m1_plus)

    svd_300 = TruncatedSVD(n_components=300)
    m2_300 = svd_300.fit_transform(m1_plus)

    # calculate similarity
    s_m1 = np.zeros_like(s)
    for v, w in p.keys():
        s_m1[p[(v, w)]] = calc_sim(m1, v, w, vocab_index)
    print(f"Pearson correlation between s and s_m1: {np.corrcoef(s, s_m1)[0, 1]}")

    s_m1_plus = np.zeros_like(s)
    for v, w in p.keys():
        s_m1_plus[p[(v, w)]] = calc_sim(m1_plus, v, w, vocab_index)
    print(f"Pearson correlation between s and s_m1_plus: {np.corrcoef(s, s_m1_plus)[0, 1]}")

    s_m2_10 = np.zeros_like(s)
    for v, w in p.keys():
        s_m2_10[p[(v, w)]] = calc_sim(m2_10, v, w, vocab_index)
    print(f"Pearson correlation between s and s_m2_10: {np.corrcoef(s, s_m2_10)[0, 1]}")

    s_m2_100 = np.zeros_like(s)
    for v, w in p.keys():
        s_m2_100[p[(v, w)]] = calc_sim(m2_100, v, w, vocab_index)
    print(f"Pearson correlation between s and s_m2_100: {np.corrcoef(s, s_m2_100)[0, 1]}")

    s_m2_300 = np.zeros_like(s)
    for v, w in p.keys():
        s_m2_300[p[(v, w)]] = calc_sim(m2_300, v, w, vocab_index)
    print(f"Pearson correlation between s and s_m2_300: {np.corrcoef(s, s_m2_300)[0, 1]}")
