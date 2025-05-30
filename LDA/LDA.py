import sys

sys.path.append(r"e:\2. work\ML_interview")

from typing import List
import numpy as np
from utils import loadDataForBBC
import time
import string
import pandas as pd
from nltk.corpus import stopwords


def load_data(file: str, K: int) -> List[str]:
    """
    INPUT:
        file : path of data
        K: setting topics
    """
    df = pd.read_csv(file)
    org_topics = df["category"].unique().tolist()
    M = df.shape[0]  # number of texts
    alpha = np.zeros(K)  # the distribution of topics
    beta = np.zeros(1000)  # the distribution of words

    for k, topic in enumerate(org_topics):
        alpha[k] = df[df["category"] == topic].shape[0] / M

    df.drop("category", axis=1, inplace=True)

    texts = []
    words = []
    for i in df["text"].values:
        ele = i.translate(
            str.maketrans("", "", string.punctuation)
        )  ## delete sentence symbols
        ele = [j for j in ele.split() if j not in stopwords.words("english")]
        ele = [
            j for j in ele if len(j) > 3
        ]  ## only consider the length of words more than 3
        texts.append(ele)
        words.extend(set(ele))
    words = list(set(words))
    words_cnt = np.zeros(len(words))  ## store the occurance of words

    for i in range(len(texts)):
        ele = texts[i]
        for word in ele:
            ind = words.index(word)  ## index store in words_cnt
            words_cnt[ind] += 1

    sort_idxs = np.argsort(words_cnt)[::-1]  ## descedning order
    words = [words[idx] for idx in sort_idxs[:1000]]

    for i in range(len(texts)):
        ele = []
        for word in texts[i]:
            if word in words:
                idx = words.index(word)
                ele.append(word)
                beta[idx] += 1  ## statistics the occurrences of each word
        texts[i] = ele
    beta /= np.sum(beta)

    return org_topics, texts, words, alpha, beta


class LDA:
    def __init__(self) -> None:
        pass

    def do_lda(self, texts, words, alpha, beta, K, iters):
        M = len(texts)
        V = len(words)
        N_MK = np.zeros((M, K))  ## text-topic matrix
        N_KV = np.zeros((K, V))  ## topic-words matrix
        N_M = np.zeros(M)  ## count vector of texts
        N_K = np.zeros(K)  ## count vector of topics
        ## 20.2
        Z_MN = []
        for m in M:
            zm = []
            ele = texts[m]
            for idx, word in enumerate(ele):
                v = words.index(word)
                z = np.random.randint(K)
                zm.append(z)
                N_MK[m, z] += 1
                N_M[m] += 1
                N_KV[z, v] += 1
                N_K[z] += 1
            Z_MN.append(zm)

        ## begin to iteration
        ### 20.3
        for i in range(iters):
            print("{}/{}".format(i + 1, iters))
            for m in range(M):
                ele = texts[m]
                for idx, word in enumerate(ele):
                    v = words.index(word)
                    z = Z_MN[m][idx]
                    N_MK[m, z] -= 1
                    N_M[m] -= 1
                    N_KV[z][v] -= 1
                    N_K[z] -= 1

                    p = []
                    sums_k = 0
                    for k in range(K):
                        p_zk = (N_KV[k][v] + beta[v]) * (N_MK[m][k] + alpha[k])
                        sums_v = 0
                        sums_k += N_MK[m][k] + alpha[k]
                        for w in range(V):
                            sums_v += N_KV[k][w] + beta[w]
                        p_zk /= sums_v
                        p.append(p_zk)
                    p = p / sums_k
                    p = p / np.sum(p)
                    new_z = np.random.choice(a=K, p=p)
                    Z_MN[m][idx] = new_z
                    N_MK[m, new_z] += 1
                    N_M[m] += 1
                    N_KV[new_z, v] += 1
                    N_K[new_z] += 1

        # 20.4
        theta = np.zeros((M, K))
        phi = np.zeros((K, V))
        for m in range(M):
            sums_k = 0
            for k in range(K):
                theta[m, k] = N_MK[m][k] + alpha[k]
                sums_k += theta[m, k]
            theta[m] /= sums_k
        for k in range(K):
            sums_v = 0
            for v in range(V):
                phi[k, v] = N_KV[k][v] + beta[v]
                sums_v += phi[k][v]

            phi[k] /= sums_v

        return theta, phi


if __name__ == "__main__":
    K = 5
    org_topics, text, words, alpha, beta = load_data("e:/ML_data/bbc_text.csv", K)
    print("Original Topics:")
    print(org_topics)
    start = time.time()
    iters = 2
    lda = LDA()
    theta, phi = lda.do_lda(text, words, alpha, beta, K, iters)

    for k in range(K):
        sort_inds = np.argsort(phi[k])[::-1]
        topic = []
        for i in range(10):
            topic.append(words[sort_inds[i]])
        topic = " ".join(topic)
        print("Topic {}: {}".format(k + 1, topic))
    end = time.time()
    print("Time:", end - start)
