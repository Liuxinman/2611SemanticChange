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
num_semantic_full = 0
full_semantic = open("data/full_semantic.txt", "w")
num_syntactic_full = 0
full_syntactic = open("data/full_syntactic.txt", "w")
num_semantic = 0
semantic_test_f = open("data/semantic_test.txt", "w")
num_syntactic = 0
syntactic_test_f = open("data/syntactic_test.txt", "w")
semantic = False
for line in word_test_f.readlines():
    if line[0] == ":":
        small_test_f.writelines(line)
        if line[2:6] == "gram":
            syntactic_test_f.writelines(line)
            full_syntactic.writelines(line)
            semantic = False
        else:
            semantic_test_f.writelines(line)
            full_semantic.writelines(line)
            semantic = True
    else:
        words = line.split()
        save_to_small = words[0] in vocab_idx and words[1] in vocab_idx and words[2] in vocab_idx
        if save_to_small:
            small_test_f.writelines(line)
        if semantic:
            num_semantic_full += 1
            full_semantic.writelines(line)
            if save_to_small:
                semantic_test_f.writelines(line)
                num_semantic += 1
        else:
            num_syntactic_full += 1
            full_syntactic.writelines(line)
            if save_to_small:
                syntactic_test_f.writelines(line)
                num_syntactic += 1

print("------------ Full dataset ------------")
print(f"the number of all samples: {num_semantic_full + num_syntactic_full}.")
print(f"the number of semantic samples: {num_semantic_full}.")
print(f"the number of syntactic samples: {num_syntactic_full}.")

print("------------ Small dataset ------------")
print(f"the number of all samples: {num_semantic + num_syntactic}.")
print(f"the number of semantic samples: {num_semantic}.")
print(f"the number of syntactic samples: {num_syntactic}.")

word_test_f.close()
small_test_f.close()
semantic_test_f.close()
syntactic_test_f.close()
full_semantic.close()
full_syntactic.close()

# test on full dataset
result = w2v.evaluate_word_analogies("data/word-test.v1.txt")

print(f"semantic analogy test and syntactic analogy test accuracy - full dataset:")
print(f"word2vec: {result[0]}")

result = w2v.evaluate_word_analogies("data/full_semantic.txt")

print(f"semantic analogy test accuracy - full dataset:")
print(f"word2vec: {result[0]}")

result = w2v.evaluate_word_analogies("data/full_syntactic.txt")

print(f"syntactic analogy test accuracy - full dataset:")
print(f"word2vec: {result[0]}")

# test on a smaller set
w2v_results = w2v.evaluate_word_analogies("data/small_test_set.txt")
lsa_results = lsa_300.evaluate_word_analogies("data/small_test_set.txt")
print(f"semantic analogy test and syntactic analogy test accuracy - small dataset:")
print(f"lsa 300: {lsa_results[0]}")
print(f"word2vec: {w2v_results[0]}")

w2v_results = w2v.evaluate_word_analogies("data/semantic_test.txt")
lsa_results = lsa_300.evaluate_word_analogies("data/semantic_test.txt")
print(f"semantic analogy accuracy - small dataset:")
print(f"lsa 300: {lsa_results[0]}")
print(f"word2vec: {w2v_results[0]}")

w2v_results = w2v.evaluate_word_analogies("data/syntactic_test.txt")
lsa_results = lsa_300.evaluate_word_analogies("data/syntactic_test.txt")
print(f"syntactic test accuracy - small dataset:")
print(f"lsa 300: {lsa_results[0]}")
print(f"word2vec: {w2v_results[0]}")
