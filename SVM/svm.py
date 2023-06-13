import sys
sys.path.append(r'e:\2. work\ML_interview')

import numpy as np
import math
import random
from typing import List, Dict, Optional, Union
from utils import loadDataForMniST

class SVM:
    def __init__(
                self,
                trainDataList: List[List[int]],
                trainLabelList: List[int],
                sigma: int = 10,
                C : int = 200,
                toler: float = 0.001
                ) -> None:
        self.trainDataArr = np.mat(trainDataList)
        self.trainLabelArr = np.mat(trainLabelList).T
        # self.m = self.trainDataArr.shape[0]
        # self.n = self.trainDataArr.shape[1]
        self.m, self.n = np.shape(self.trainDataArr)
        self.sigma = sigma  
        self.C = C
        self.toler = toler
        self.kernel = self._computeKernel()
        self.b = 0
        self.alpha = [0] * self.trainDataArr.shape[0]
        self.E = [0 * self.trainLabelArr[i, 0] for i in range(self.trainLabelArr.shape[0])]
        self.supportVecIndex = []
    
    def _computeKernel(self):
        '''
            computer kernel function
            return: gaussian kernel matrix
        '''
        kernel = [[0 for i in range(self.m)] for j in range(self.m)]

        for i in range(self.m):
            X = self.trainDataArr[i,:]
            for j in range(i, self.m):
                Z = self.trainDataArr[j,:]
                gauskernelEle = np.exp(-1 * (X - Z) * (X - Z).T / (2 * self.sigma ** 2))
                kernel[i][j] = gauskernelEle
                kernel[j][i] = gauskernelEle

        return kernel

    def _calc_gxi(self, i) -> np.ndarray:
        gxi = 0
        ## record non-zero index
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.trainLabelArr[j] * self.kernel[j][i]
        
        gxi += self.b
        return gxi

    def _calc_Ei(self, i:int) -> np.ndarray:
        gxi = self._calc_gxi(i)
        return gxi -self.trainLabelArr[i]

    def _calcSingerKernel(self, x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
        result = np.exp(-1 *  (x1 - x2) * (x1 - x2).T / (2 * self.sigma ** 2))
        return np.exp(result)  
    
    def _isKKT(self, i:int) ->bool:
        '''
            justify each point if it satisify the KKT condition
        '''
        gxi = self._calc_gxi(i)
        yi = self.trainLabelArr[i]
        ## softSVM KKT
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True
        
        return False
    
    
    def _getAlphaJ(self, E1:np.ndarray, i:int) -> Union[np.ndarray, int]:
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1

        nonZeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        for j in nonZeroE:
            E2_tmp = self._calc_Ei(j)
            if math.fabs(E1-E2_tmp) > maxE1_E2:
                maxE1_E2 = E1 - E2_tmp
                E2 = E2_tmp
                maxIndex = j
            
            if maxIndex == -1:
                maxIndex = i
                while maxIndex == i:
                    maxIndex = int(random.uniform(0, self.m))
                
                E2 = self._calc_Ei(maxIndex)

        return E2, maxIndex

    
    def train(self, iter:int=100):
        step = 0
        parameterChanged = 1
        while (step < iter)  and (parameterChanged > 0):
            print('iter:%d:%d'%( step, iter))
            step += 1
            parameterChanged = 0

            for i in range(self.m):
                if self._isKKT(i) == False:
                    E1 = self._calc_Ei(i)
                    E2, j = self._getAlphaJ(E1,i)

                    y1 = self.trainLabelArr[i]
                    y2 = self.trainLabelArr[j]

                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    if L == H:   continue

                    k11 = self.kernel[i][i]
                    k22 = self.kernel[j][j]
                    k21 = self.kernel[j][i]
                    k12 = self.kernel[i][j]


                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    if alphaNew_2 < L: alphaNew_2 = L
                    elif alphaNew_2 > H: alphaNew_2 = H
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New

                    else:
                        bNew = (b1New + b2New) / 2
                    
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self._calc_Ei(i)
                    self.E[j] = self._calc_Ei(j)

                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1
                    
                    print("iter: %d i:%d, pairs changed %d" % (step, i, parameterChanged))
        
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)
   

    def _predict(self, x:np.ndarray) -> np.ndarray:
        result = 0
        for i in self.supportVecIndex:
            tmp = self._calcSingerKernel(self.trainDataArr[i, :], np.mat(x))
            result += self.alpha[i] * self.trainLabelArr[i] * tmp

        result += self.b

        return np.sign(result)
    

    def test(self, testDataList: List[List[int]], testLabelList:List[int]) -> float:
        correctCnt = 0
        for i in range(len(testDataList)):
            print('test:%d:%d'%(i, len(testDataList)))
            result = self._predict(testDataList[i])
            if result == testLabelList[i]:
                correctCnt += 1
        return correctCnt / len(testDataList)
    



if __name__ == '__main__':
    import time
 

    start = time.time()

    print('start read trainsSet')
    trainDataList, trainLabelList = loadDataForMniST('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadDataForMniST('e:\ML_data\mnist_test.csv')

    print('start init SVM')
    svm = SVM(trainDataList[:10], trainLabelList[:10], 10, 200, 0.001)

    print('start to train:')
    svm.train()

    print('start to test')
    accuracy = svm.test(testDataList[:10], testLabelList[:10])
    print('the accuracy is:%d'%(accuracy * 100), '%')

    print('time span:', time.time() - start)