import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


# Step 2: extract embeddings
w2v = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)

# Step 3: calc cos similarity and report pearson correlation
with open("data/p.plk", "rb") as f:
    p = pickle.load(f)
with open("data/s.plk", "rb") as f:
    s = pickle.load(f)

s_word2vec = np.zeros_like(s)
for v, w in p.keys():
    s_word2vec[p[(v, w)]] = cosine_similarity(w2v[v].reshape(1, -1), w2v[w].reshape(1, -1))[0][0]

print(f"Pearson correlation between s and s_word2vec: {pearsonr(s, s_word2vec)}")

# Step 4: analogy test
result = w2v.evaluate_word_analogies("data/word-test.v1.txt")

print(f"semantic analogy test and semantic syntactic test accuracy - full dataset:")
print(f"word2vec: {result[0]}")

# create lsa 300 model
with open("data/lsa_m2_300.plk", "rb") as f:
    m2_300 = pickle.load(f)

with open("data/vocab.plk", "rb") as f:
    vocab_idx = pickle.load(f)

lsa_300 = KeyedVectors(m2_300.shape[1])
lsa_300.add_vectors(list(vocab_idx.keys()), m2_300)

# generate smaller dataset
word_test_f = open("data/word-test.v1.txt", "r")
small_test_f = open("data/small_test_set.txt", "w")
for line in word_test_f.readlines():
    if line[0] == ":":
        small_test_f.writelines(line)
    else:
        words = line.split()
        if words[0] in vocab_idx and words[1] in vocab_idx and words[2] in vocab_idx:
            small_test_f.writelines(line)
word_test_f.close()
small_test_f.close()

# test on a smaller set
w2v_results = w2v.evaluate_word_analogies("data/small_test_set.txt")
lsa_results = lsa_300.evaluate_word_analogies("data/small_test_set.txt")

print(f"semantic analogy test and semantic syntactic test accuracy - small dataset:")
print(f"lsa 300: {lsa_results[0]}")
print(f"word2vec: {w2v_results[0]}")
