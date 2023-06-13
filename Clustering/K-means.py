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

class KMeans:
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

    def _cacul_clusterCenter(self, cluster: List, dataArr:np.ndarray) -> np.ndarray:
        '''
         caculate the position of center for each cluster
         : cluster: all data points that the current cluster includes
         : return: the new center
        '''
        center = np.zeros(dataArr.shape[1])
        for i in range(dataArr.shape[1]):
            for n in cluster:
                center[i] += dataArr[n][i]   # caculate the sum of the i-th features in current cluster
        center = center / dataArr.shape[0]   # # caculate the mean for each feature

        return center

    def _Adjusted_Rand_Index(self, cluster_dict: Dict, labels:List[int], k:int) -> Union[np.ndarray, float]:
        '''
          :: reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.htmld
        '''
        group_array = np.zeros((k, k))
        labelSet = list(set(labels))
        y_dict = {i:[] for i in range(k)}

        for i in range(len(labels)):
            y_dict[labelSet.index(labels[i])].append(i)
        
        # group_array = [group_array[i][j] += 1 for i in range(k) for j in range(k) for n in range(len(labels)) if n in group_dict[i] and n in y_dict[j]]
        for i in range(k):
            for j in range(k):
                for n in range(len(labels)):
                    if n in cluster_dict[i] and n in y_dict[j]:
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
    
    def kmeans(self, dataArr: np.ndarray, k: int, iter_steps: int, labels: List[int]) -> Union[Dict, List]:
        
        centers_idx = random.sample(range(dataArr.shape[0]), k)
        centers = [dataArr[ci] for ci in centers_idx]
        scores = []

        #### !!! 
        for i in range(iter_steps):
            cluster_dict = {i:[] for i in range(k)}
            print('{}/{}'.format(i+1, iters))
            # caculate distance
            ### cacluate the distance between every center of cluster and every data instance
            for j in range(dataArr.shape[0]):
                dists = []
                for ci in range(k):
                    dist = self._cacul_distance(dataArr[j], centers[ci])
                    dists.append(dist)
                minCluster_idx = dists.index(min(dists))
                ## store the j-th data into the list corresponds to the min_cluster_index
                cluster_dict[minCluster_idx].append(j)
            ## according to data in a cluster, recaculate the cluster center.
            for n in range(k):
                centers[n] = self._cacul_clusterCenter(cluster_dict[n], dataArr)
            
            scores.append(self._Adjusted_Rand_Index(cluster_dict, labels,k))

        return cluster_dict, scores


if __name__ == "__main__":
    Xarray, labelList = loadDataForIris('e:\ML_data\iris.data')
    start = time.time()  
    
    model = KMeans()
    Xarray = model.Nomalize(Xarray) 
    k = 3 
    iters = 2
    group_dict, scores = model.kmeans(Xarray, k, iters, labels=labelList)  
    end = time.time()  
    print(group_dict)
    print('Time:', end-start)
    print("Scores:", scores)
    plt.plot(range(iters), scores)  
    plt.show()
