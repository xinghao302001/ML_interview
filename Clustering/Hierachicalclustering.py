import sys
sys.path.append(r'e:\2. work\ML_interview')

import numpy as np
import time
import random 
from scipy.special import comb
import matplotlib.pyplot as plt
import math
from typing import List, Optional, Dict, Union
from utils import loadDataForIris


class Hierachicalclustering:
    def __init__(self) -> None:
        pass

    def Nomalize(self,
                  dataArray: np.ndarray
                  ):
        '''
            feature noramlization
            : return: min-max scalling
        '''
        for feature in range(dataArray.shape[1]):
            maxfeature = np.max(dataArray[:, feature])
            minfeature = np.min(dataArray[:, feature])
            for i in range(dataArray.shape[0]):
                dataArray[i][feature] = (dataArray[i][feature] - minfeature) / (maxfeature-minfeature)
        return dataArray

    def _cacul_distance(self, sample_a: np.ndarray, sample_j:np.ndarray) -> float:
        dist = 0
        for col in range(len(sample_a)):
            dist += (sample_a[col] - sample_j[col]) ** 2
        dist = math.sqrt(dist)
        return dist
    
    def _Adjusted_Rand_Index(self, cluster_dict: Dict, labels:List[int], k:int) -> Union[np.ndarray, float]:
        group_array = np.zeros((k, k)) 
        
        y_dict = {}

        for i in range(len(labels)):
            if labels[i] not in y_dict:
                y_dict[labels[i]] = [i]
            else:
                y_dict[labels[i]].append(i)
        
        # group_array = [group_array[i][j] += 1 for i in range(k) for j in range(k) for n in range(len(labels)) if n in group_dict[i] and n in y_dict[j]]
        for i in range(k):
            for j in range(k):
                for n in range(len(labels)):
                    if n in group_dict[list(group_dict.keys())[i]] and n in y_dict[list(y_dict.keys())[j]]:
                        group_array[i][j] += 1
        RI = 0 # init Rand Index
        sum_i = np.zeros(k)
        sum_j = np.zeros(k)

        for i in range(k):
            for j in range(k):
                sum_i[i] += group_array[i][j]
                sum_j[j] += group_array[i][j]
                if group_array[i][j] >= 2:
                    RI += comb(group_array[i][j], 2)
        ci = 0
        cj = 0

        for i in range(k):
            if sum_i[i] >= 2:
                ci += comb(sum_i[i], 2)
        for j in range(k):
            if sum_j[j] >= 2:
                cj += comb(sum_j[j], 2)
                
        E_RI = ci * cj / comb(len(labels), 2)  
        max_RI = (ci + cj) / 2  
    
        return (RI-E_RI) / (max_RI-E_RI)  
    
    def allDistances(self, dataArr:np.ndarray) -> np.ndarray:
        dists = np.zeros((dataArr.shape[0], dataArr.shape[0]))
        for n1 in range(dataArr.shape[0]):
            for n2 in range(n1):
                dists[n1][n2] = self._cacul_distance(dataArr[n1], dataArr[n2])
                dists[n2][n1] = dists[n1][n2]
        return dists
    
    def _cal_clusterList(self, group_a: int, group_b: int, cluster_dict: Dict, dists: np.ndarray):

        d = []
        for xi in cluster_dict[group_a]:
            for xj in cluster_dict[group_b]:
                if xi != xj:
                    d.append(dists[xi][xj])

        return min(d)
    
    def Clustering(self, dataArr:np.ndarray, k: int, dists: np.ndarray):
        group_dict = {}
        for n in range(dataArr.shape[0]):
            group_dict[n] = [n]
        
        NewGroup = dataArr.shape[0]

        while len(group_dict.keys()) > k:
            print('Number of groups:', len(group_dict.keys()))
            group_dists = {}

            for g1 in group_dict.keys():
                for g2 in group_dict.keys():
                    if g1 != g2:
                        if (g1, g2) not in group_dists.values():
                            d = self._cal_clusterList(g1, g2, group_dict, dists)
                            group_dists[d] = (g1, g2)

            group_mindist = min(list(group_dists.keys()))
            min_groups = group_dists[group_mindist]

            new_group = []
            for g in min_groups:
                new_group.extend(group_dict[g])
                del group_dict[g]
            print(NewGroup, new_group)
            group_dict[NewGroup] = new_group

            NewGroup += 1
        
        return group_dict





if __name__ == "__main__":
    Xarray, labelList = loadDataForIris('e:\ML_data\iris.data')
    start = time.time()  
    
    model = Hierachicalclustering()
    Xarray = model.Nomalize(Xarray)
    dists = model.allDistances(Xarray) 
    k = 3 
    group_dict = model.Clustering(Xarray, k, dists)  
    end = time.time()  
    print(group_dict)
    ARI = model._Adjusted_Rand_Index(group_dict,labelList, k)
    print('Adjusted Rand Index:', ARI)
    print('Time:', end-start)