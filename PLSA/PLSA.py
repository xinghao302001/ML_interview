import sys
sys.path.append(r'e:\2. work\ML_interview')

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import time
from utils import loadDataForBBC
from typing import List, Union

class PLSA:
    
    def __init__(self) -> None:
        pass

    def freq_counter(texts:List[List[str]], words:List[str]) -> Union[np.ndarray, List]:
        
        words_cnt = np.zeros(len(words))
        X = np.zeros((100, len(texts))) ## m*n: word-text co-occurance matrix -> set m=300 to avoid excution time becoming too longer
        
        for i in range(len(texts)):
            text = texts[i]
            for word in text:
                idx = words.index(word)
                words_cnt[idx] += 1
        
        sorted_idxs = np.argsort(words_cnt)[::-1]
        ## store the words whose occurance times is more than 1000 
        words = [words[idx] for idx in sorted_idxs[:1000]]

        for i in range(len(texts)):
            text = texts[i]
            for word in text:
                if word in words:
                    idx = words.index(word)
                    X[idx, i] += 1

        return words, X


    def plsa(X: np.ndarray, K:int, words: List[str],iters:int=10) -> np.ndarray:
        '''
            INPUT:
            X - (array) words-text occurance matrix
            K - (int)  # of topics
            words - (list) 
            iters - (int) 
            
            OUTPUT:
            P_wi_zk - (array) the occurance probility of i-th word given topic z_k
            P_zk_dj - (array) the occurance probility of k-th topic given document d_j
            
        '''
        # M - # of words. N: # of texts
        M, N = X.shape

        P_wi_zk = np.random.rand(K,M)
        for k in range(K):
            P_wi_zk[k] /= np.sum(P_wi_zk)

        P_zk_dj = np.random.rand(N, K)
        for n in range(N):
            P_zk_dj[n] /= np.sum(P_zk_dj[n])

        P_zk_wi_dj = np.zeros((M, N, K))

        ## EM algorithms
        for i in range(iters):
            print('{}/{}'.format(i+1, iters))
            # E step
            for m in range(M):
                for n in range(N):
                    sums = 0
                    for k in range(K):
                        P_zk_wi_dj[m, n ,k] = P_wi_zk[k, m] * P_zk_dj[n,k]
                        sums += P_zk_wi_dj[m, n, k]
                    ### !!! 
                    P_zk_wi_dj = P_zk_wi_dj[m, n, :] / sums

            # M step
            ## caculate P_wi_zk
            for k in range(K):
                sum_deno = 0
                for m in range(M):
                    P_wi_zk[k, m] = 0
                    for n in range(N):
                        P_wi_zk[k, m] += X[m, n] * P_zk_wi_dj[m, n, k]
                    sum_deno += P_wi_zk[k, m]
                P_wi_zk[k, :] = P_wi_zk[k, :] / sum_deno

            ## caculate P_zk_dj
            for n in range(N):
                for k in range(k):
                    P_zk_dj[n, k] = 0
                    for m in range(M):
                        P_zk_dj[n, k] += X[m,n] * P_zk_wi_dj[m, n, k]
                    P_zk_dj[n, k] = P_zk_dj / np.sum(X[:, n])
        
        return P_wi_zk, P_zk_dj
    

if __name__ == "__main__":
    org_topics, text, words = loadDataForBBC('e:/ML_data/bbc_text.csv')
    print('Original Topics:')
    print(org_topics)  
    start = time.time() 
    model = PLSA()

    X = model.freq_counter(text, words)  
    k = 5  
    topics = model.lsa(X, k, words) 
    print('Generated Topics:')
    for i in range(k):
        print('Topic {}: {}'.format(i+1, topics[i])) 
    end = time.time()  
    print('Time:', end-start)
