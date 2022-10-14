import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download("brown")
from nltk.corpus import brown

import string
import pickle
import math
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def contain_punc(word):
    punctuation = [p for p in string.punctuation]
    for p in punctuation:
        if p in word:
            return True
    return False


def generate_vocab():
    stop_words = stopwords.words("english")
    text = [
        w.lower() for w in brown.words() if not contain_punc(w) and w not in stop_words
    ]  # remove punc and stopwords
    fDist = nltk.FreqDist(text)
    fDist_5000 = fDist.most_common(5000)
    print("the 5 most common words: \n", fDist_5000[:5])
    print("the 5 least common words: \n", fDist_5000[-5:])

    with open("table1_sim.plk", "rb") as f:
        table1_sim = pickle.load(f)

    table1 = []
    for t in table1_sim.keys():
        table1.append(t[0])
        table1.append(t[1])
    
    fDist_5000_w = set(list(zip(*fDist_5000))[0])

    vocab = list(fDist_5000_w.union(set(table1)))
    print(f"|W| = {len(vocab)}.\n")

    return vocab


if __name__ == "__main__":
    # Step 2: extract the 5000 most common words
    vocab = generate_vocab()
    vocab_index = {vocab[i]: i for i in range(len(vocab))}

    # Step 3: word-context matrix
    corpus = brown.words()
    m1 = np.zeros((len(vocab), len(vocab)))
    for i in range(len(corpus) - 1):
        cur_w = corpus[i].lower()
        next_w = corpus[i + 1].lower()
        if cur_w in vocab_index and next_w in vocab_index:
            m1[vocab_index[cur_w], vocab_index[next_w]] += 1
            if next_w != cur_w:
                m1[vocab_index[next_w], vocab_index[cur_w]] += 1

    # Step4: PPMI
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

    # Step 5: Latent Semantic Analysis
    svd_10 = PCA(n_components=10)
    m2_10 = svd_10.fit_transform(m1_plus)

    svd_100 = PCA(n_components=100)
    m2_100 = svd_100.fit_transform(m1_plus)

    svd_300 = PCA(n_components=300)
    m2_300 = svd_300.fit_transform(m1_plus)

    # Step 6: find the intersection between W and table1 - P and S
    with open("table1_sim.plk", "rb") as f:
        table1_sim = pickle.load(f)

    i = 0
    s = []
    p = {}
    for key in table1_sim.keys():
        v, w = key
        if v in vocab and w in vocab:
            p[(v, w)] = i
            s.append(table1_sim[key])
            i += 1
    
    with open("p.plk", "wb") as f:
        pickle.dump(p, f)
    with open("s.plk", "wb") as f:
        pickle.dump(s, f)

    # Step 7: calculate similarity
    s_m1 = np.zeros_like(s)
    s_m1_plus = np.zeros_like(s)
    s_m2_10 = np.zeros_like(s)
    s_m2_100 = np.zeros_like(s)
    s_m2_300 = np.zeros_like(s)
    for v, w in p.keys():
        s_m1[p[(v, w)]] = cosine_similarity(
            m1[vocab_index[v]].reshape(1, -1), m1[vocab_index[w]].reshape(1, -1)
        )[0][0]
        s_m1_plus[p[(v, w)]] = cosine_similarity(
            m1_plus[vocab_index[v]].reshape(1, -1), m1_plus[vocab_index[w]].reshape(1, -1)
        )[0][0]
        s_m2_10[p[(v, w)]] = cosine_similarity(
            m2_10[vocab_index[v]].reshape(1, -1), m2_10[vocab_index[w]].reshape(1, -1)
        )[0][0]
        s_m2_100[p[(v, w)]] = cosine_similarity(
            m2_100[vocab_index[v]].reshape(1, -1), m2_100[vocab_index[w]].reshape(1, -1)
        )[0][0]
        s_m2_300[p[(v, w)]] = cosine_similarity(
            m2_300[vocab_index[v]].reshape(1, -1), m2_300[vocab_index[w]].reshape(1, -1)
        )[0][0]

    # Step 8: report pearson correlation
    print(f"Pearson correlation between s and s_m1: {pearsonr(s, s_m1)}")
    print(f"Pearson correlation between s and s_m1_plus: {pearsonr(s, s_m1_plus)}")
    print(f"Pearson correlation between s and s_m2_10: {pearsonr(s, s_m2_10)}")
    print(f"Pearson correlation between s and s_m2_100: {pearsonr(s, s_m2_100)}")
    print(f"Pearson correlation between s and s_m2_300: {pearsonr(s, s_m2_300)}")
