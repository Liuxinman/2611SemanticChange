import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


# Step 1
with open("embeddings/data.pkl", "rb") as f:
    data = pickle.load(f)

word = data["w"]
emb = data["E"]  # w x d x #features
# remove noisy data
valid_idx = []
word_removed = []
for i in range(len(emb)):
    zero_emb = False
    for j in range(len(emb[0])):
        if sum(emb[i][j]) == 0:
            zero_emb = True
            word_removed.append(word[i])
            break
    if not zero_emb:
        valid_idx.append(i)
emb = np.array(emb)[valid_idx]
print(emb.shape)
print(f"noisy word removed: {word_removed}")

# Step 2
# Method 1: trivial way
# compute the cosine distance between the word emb of first time interval
# and word emb of last time interval
first = emb[:, 0, :]
last = emb[:, -1, :]

sim_score = np.diagonal(1 - cosine_similarity(first, last))
result1 = sorted(list(zip(word, sim_score)), key=lambda x: x[1])

print("--------- Method 1: compute cosine sim between first and last interval ---------")
print("Top 20 least changing words:")
print(", ".join([w[0] for w in result1[:20]]))
print("Top 20 most changing words:")
print(", ".join([result1[i][0] for i in range(-1, -21, -1)]))
print("\n")

# Method 2:
# compute the maximum cosine distance between adjacent decades
all_sim1 = np.zeros((emb.shape[0], emb.shape[1] - 1))
for i in range(emb.shape[1] - 1):
    cur = emb[:, i, :]
    next = emb[:, i + 1, :]
    all_sim1[:, i] = np.diagonal(1 - cosine_similarity(cur, next))
max_sim_score1 = np.amax(all_sim1, axis=1)
result2 = sorted(list(zip(word, max_sim_score1)), key=lambda x: x[1])

print("--------- Method 2: compute the maximum cosine distance between adjacent decades ---------")
print("Top 20 least changing words:")
print(", ".join([w[0] for w in result2[:20]]))
print("Top 20 most changing words:")
print(", ".join([result2[i][0] for i in range(-1, -21, -1)]))
print("\n")

# Method 3:
# compute the maximum cosine distance between the first decades and other decades
all_sim2 = np.zeros((emb.shape[0], emb.shape[1] - 1))
for i in range(1, emb.shape[1]):
    cur = emb[:, i, :]
    all_sim2[:, i - 1] = np.diagonal(1 - cosine_similarity(first, cur))
max_sim_score2 = np.amax(all_sim2, axis=1)
result3 = sorted(list(zip(word, max_sim_score2)), key=lambda x: x[1])

print(
    "--------- Method 3:compute the maximum cosine distance between the first decades and other decades ---------"
)
print("Top 20 least changing words:")
print(", ".join([w[0] for w in result3[:20]]))
print("Top 20 most changing words:")
print(", ".join([result3[i][0] for i in range(-1, -21, -1)]))
print("\n")

# measure intercorrelations
all_result = [sim_score, max_sim_score1, max_sim_score2]
pearson = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        pearson[i, j] = pearsonr(all_result[i], all_result[j])[0]
print("--------- Intercorrelations ---------")
print(pearson)
print("\n")


# Step 3: Evaluation
# compute the overlap of k nearest neighbour between first and last interval
K = 50
overlap = np.zeros(emb.shape[0])
sim_first = cosine_similarity(first, first)
sim_last = cosine_similarity(last, last)
for i in range(emb.shape[0]):
    sim_first_i = sorted(zip(word, sim_first[:, i]), key=lambda x: x[1])[-K:]
    sim_last_i = sorted(zip(word, sim_last[:, i]), key=lambda x: x[1])[-K:]
    first_k = [w[0] for w in sim_first_i]
    last_k = [w[0] for w in sim_last_i]
    overlap[i] = len(set(first_k).intersection(set(last_k)))
overlap = (-1) * overlap
true_ranking = sorted(list(zip(word, overlap)), key=lambda x: x[1])

# measure intercorrelations
all_result = [sim_score, max_sim_score1, max_sim_score2, overlap]
pearson = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        pearson[i, j] = pearsonr(all_result[i], all_result[j])[0]

print("--------- Intercorrelations with true ranking ---------")
print(pearson)
print("\n")

# Step 4: detecting the point(s) of semantic change
top_3_word = result1[-3:]
print(f"top 3 words: {top_3_word}")
top_3_idx = [word.index(w[0]) for w in top_3_word]
top_3_sim = all_sim1[top_3_idx]
# for i in range(3):
#     slope = []
#     for j in range(top_3_sim.shape[1] - 1):
#         slope.append((abs(top_3_sim[i, j + 1] - top_3_sim[i, j]) >= 0.1))
#     print(slope)
change_point = np.array(data['d'][1:])[np.argmax(top_3_sim, axis=1)]
print(f"change point of the top 3 words: {change_point}")

x = [i for i in range(9)]
plt.plot(x, top_3_sim[0], label=top_3_word[0][0])
plt.plot(x, top_3_sim[1], label=top_3_word[1][0])
plt.plot(x, top_3_sim[2], label=top_3_word[2][0])
plt.xticks(x, data["d"][:-1])
plt.xlabel("decade")
plt.ylabel("cosine distance")
plt.legend()
plt.savefig("changeOfPoint.png")
