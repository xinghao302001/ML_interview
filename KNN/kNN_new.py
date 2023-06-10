import sys
sys.path.append(r'e:\2. work\ML_interview')
from utils import loadDataForMniST

import numpy as np
from typing import List
import time

class KNN:
    def __init__(
                  self,
                  trainDataList: List[List[int]],
                  trainLabelList: List[int],
                  k: int,
                  ) -> None:
        
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.topK = k

    def _calDist(
                self,
                sample_a: np.ndarray,
                sample_b: np.ndarray,
                ) -> float:
        '''
            caculate this distance between two samples.
            : sample_a: the vector of sample a
            : sample_b: the vector of sample b

        '''
        return np.sqrt(np.sum(np.square(sample_a - sample_b)))

    def _getClosest(
                    self,
                    trainDataMat, 
                    trainLabelMat, 
                    x
                    ):
        distList = [0] * len(trainLabelMat)
        topK = self.topK
        for i in range(len(trainDataMat)):
            sample = trainDataMat[i]
            cur_Dist = self._calDist(sample, x)
            distList[i] = cur_Dist
        
        TopK_distArr = np.argsort(np.array(distList))[:topK]
        labelList = [0] * 10
        for index in TopK_distArr:
            labelList[int(trainLabelMat[index])] += 1
        return labelList.index(max(labelList))
    
    def test(
            self,
            testDataArr, 
            testLabelArr, 
            ):
        
        trainDataMat = np.mat(self.trainDataList); trainLabelMat = np.mat(self.trainLabelList).T
        testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T

        errorCnt = 0
        for i in range(200):
 
            print('test %d:%d' % (i, 200))
  
            x = testDataMat[i]

            y = self._getClosest(trainDataMat, trainLabelMat, x)

        if y != testLabelMat[i]: errorCnt += 1

        return 1 - (errorCnt / 200)


if __name__ == "__main__":
    start = time.time()
    start = time.time()
    print('start read trainsSet')
    trainDataList, trainLabelList = loadDataForMniST('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadDataForMniST('e:\ML_data\mnist_test.csv')


    model = KNN(trainDataList, trainLabelList, 25)
    accur = model.test(testDataList, testLabelList)
    #打印正确率
    print('accur is:%d'%(accur * 100), '%')

    end = time.time()
    #显示花费时间
    print('time span:', end - start)