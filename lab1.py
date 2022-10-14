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

from gensim.models import KeyedVectors

# Step 2: extract embeddings
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Step 3: calc cos similarity and report pearson correlation
with open("p.plk", "rb") as f:
    p = pickle.load(f)
with open("s.plk", "rb") as f:
    s = pickle.load(f)

s_word2vec = np.zeros_like(s)
for v, w in p.keys():
    s_word2vec[p[(v, w)]] = cosine_similarity(
            model[v].reshape(1, -1), model[w].reshape(1, -1)
        )[0][0]

print(f"Pearson correlation between s and s_word2vec: {pearsonr(s, s_word2vec)}")

# Step 4: analogy test

