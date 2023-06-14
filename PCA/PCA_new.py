import sys
sys.path.append(r'e:\2. work\ML_interview')

import numpy as np
import time
import random 
from scipy.special import comb
import matplotlib.pyplot as plt
import math
from typing import List, Optional, Dict, Union
from utils import loadDataForCars
import pandas as pd


class PCA:
    def __init__(self) -> None:
        pass

    def normalize(self, X:np.ndarray) -> np.ndarray:
        '''
            normalize for each feature
        '''
        row, col = X.shape ## row: # of features, col: # of samples
        for i in range(row): 
            E_xi = np.mean(X[i])
            Var_xi = np.var(X[i], ddof=1)
            for j in range(col):
                X[i][j] = (X[i][j] - E_xi) / np.sqrt(Var_xi)

        return X
    

    def _cal_V(self, X:np.ndarray) -> Union[List, np.ndarray]:
        '''
            caculate eigenvalues and V-matrix for X-array
        '''

        X_new = X.T / np.sqrt(X.shape[1] - 1)
        # Sx: covariance matrix
        Sx = np.matmul(X_new.T, X_new)
        df_Sx = pd.DataFrame(Sx)
        df_Sx[np.isnan(df_Sx)] = 3.14e-8
        V_T = []

        w, v = np.linalg.eig(Sx) # caculate eigenvalues w, and its eigenvectors v
        tmp = {} # key: eigenvalues; value: eigenvectors
        for i in range(len(w)):
            tmp[w[i]] = v[i]
        
        sorted_eigenValues = sorted(tmp, reverse=True)
        for i in sorted_eigenValues:
            d = 0
            for j in range(len(tmp[i])):
                d += tmp[i][j] ** 2
                ## caculate the unit eigenvector
            V_T.append(tmp[i] / np.sqrt(d))

        V = np.array(V_T).T
        return sorted_eigenValues, V


    def pca(self, X:np.ndarray, k:int) -> Union[List, np.ndarray]:
        
        eigenValues, V = self._cal_V(X)
        Vk_components = V[:,:k]
        Y = np.matmul(Vk_components.T, X)

        ## explainable bias ratio
        ##### this ratio decribes the i-th components can explain the ratio of the changeable of original data
        dimrates = [i / sum(eigenValues) for i in eigenValues[:k]]
        ## factor loads of storing main components
        fac_load = np.zeros((k, X.shape[0]))
        for i in range(k):
            for j in range(X.shape[0]):
                 ## caculate the factor loads that the k-components corresponds to the original j-th feature
                 fac_load[i][j] = np.sqrt(eigenValues[i]) * Vk_components[j][i] / np.sqrt(np.var(X[j]))  #计算主成分i对应原始特征j的因子负荷量，保存到fac_load中

        return fac_load, dimrates, Y


if __name__ == "__main__":
    df, X = loadDataForCars('e:\ML_data\cars.csv')  #加载数据
    start = time.time() 
    model = PCA()
    X = model.normalize(X)
    k = 3  
    fac_load, dimrates, Y = model.pca(X, k)  
    pca_result = pd.DataFrame(fac_load, index=['Dimension1', 'Dimension2', 'Dimension3'], columns=df.columns)
    pca_result['Explained Variance'] = dimrates
    pca_result.to_csv("..\PCA_results.csv")
    end = time.time() 
    print('Time:', end-start)

