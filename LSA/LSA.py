import sys
sys.path.append(r'e:\2. work\ML_interview')

from typing import List
import numpy as np
from utils import loadDataForBBC
import time

class LSA:
    def __init__(self) -> None:
        pass

    def freq_counter(texts:List[List[str]], words:List[str]) -> np.ndarray:

        X = np.zeros((len(words), len(texts)))

        for i in range(len(texts)):
            text = texts[i]
            for word in text:
                idx = words.index(word)
                X[idx][i] += 1
        
        return X
    

    def lsa(X: np.ndarray, k:int, words: List[str]) -> List[str]:
        '''
        : k: # of topics
        '''
        ## Step 1: SVD
        eigenvalue, v = np.linalg.eig(np.matmul(X.T, X))
        sorted_idxs = np.argsort(eigenvalue)[::-1]
        eigenvalue = np.sort(eigenvalue)[::-1]
        V_T = []
        for idx in sorted_idxs:
            V_T.append(v[idx]/np.linalg.norm(v[idx]))
        
        V_T = np.array(V_T)
        Sigma = np.diag(np.sqrt(eigenvalue))
        U = np.zeros((len(words), k))

        for i in range(k):
            ui = np.matmul(X, V_T.T[:,i]) / Sigma[i][i]
            U[:,i] = ui
        
        topics = []
        for i in range(k):
            idxs = np.argsort(U[:, i])[::-1]
            topic = []
            for j in range(10):
                topic.append(words[idxs[j]])
            topics.append(' '.join(topic))
        return topics



if __name__ == "__main__":
    org_topics, text, words = loadDataForBBC('e:/ML_data/bbc_text.csv')
    print('Original Topics:')
    print(org_topics)  
    start = time.time() 
    model = LSA()

    X = model.freq_counter(text, words)  
    k = 5  
    topics = model.lsa(X, k, words) 
    print('Generated Topics:')
    for i in range(k):
        print('Topic {}: {}'.format(i+1, topics[i])) 
    end = time.time()  
    print('Time:', end-start)