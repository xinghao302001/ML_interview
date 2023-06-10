import sys
sys.path.append(r'e:\2. work\ML_interview')
from utils import loadDataForMniST

import numpy as np
import time
from tqdm import tqdm
from typing import List


class NaiveBayes:
    def __init__(
                  self,
                  trainDataList: List[List[int]],
                  trainLabelList: List[int],
                  testDataList: List[List[int]],
                  testLabelList: List[int]
                  ) -> None:
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.testDataList = testDataList
        self.testLabelList = testLabelList

    
    def _naiveBayes(
                self,
                Py: float,
                Px_y: float,
                x: np.ndarray
            ) -> int:
        
        featureNum = 784
        classNum = 10
        P = [0] * classNum
        # for each class, caculate its probability
        for i in range(classNum):
            # sum of log probability
            sum = 0
            for j in range(featureNum):
                # !!!
                sum += Px_y[i][j][x[j]]
            P[i] = sum + Py[i]

        return P.index(max(P))
    
    def getProbability(
                        self,
                        dataArr: List[List[int]],
                        labelArr:List[int],
                    ) -> np.ndarray:
        '''
         get prior probability P(y) and get conditional probability P(x|y)
        '''
        featureNum = 784
        classNum = 10

        ## step 1: caculate prior probability Py
        Py = np.zeros((classNum, 1))
        for i in range(classNum):
            label_map = np.mat(dataArr) == i
            label_sum = np.sum(label_map)
            ## smoothing handeling, avoid "0-frequency" problem
            Py[i] = (label_sum + 1) / (len(dataArr) + 10)
        
        Py = np.log(Py)


        ## step 2: caculate likelihood/conditional probability Px_y = P(X=x|Y = y）
        Px_y = np.zeros((classNum, featureNum, 2))
        for i in tqdm(range(len(dataArr))):
            label = labelArr[i]
            features= dataArr[i]
            ### !!!!
            for j in range(featureNum):
                Px_y[label][j][features[j]] += 1

        ### Px_y0: in class y, the numbers of feature x_j that is chosen(i.e, 1), otherwise the feature x_j \
        ###        that is not chosen (i.e., 0)
        for label in range(classNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))
        
        return Py, Px_y
    
    def test(
            self,
            Py:float, 
            Px_y: float,
            dataArr: List[List[int]],
            labelArr: List[int]
            ) -> float:
        
            correctCnt = 0
            for i in range(len(dataArr)):
                predict = self._naiveBayes(Py, Px_y, dataArr[i])
                if predict == labelArr[i]:
                    correctCnt += 1
            return correctCnt / len(dataArr)




if __name__ == "__main__":
    start = time.time()
    print('start read trainsSet')
    trainDataList, trainLabelList = loadDataForMniST('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadDataForMniST('e:\ML_data\mnist_test.csv')


    ## model init
    model = NaiveBayes(trainDataList, trainLabelList, testDataList, testLabelList)
    print('start to train')
    Py, Px_y = model.getProbability(trainDataList[:100], trainLabelList[:100])
    print('start to test')
    accuracy = model.test(Py, Px_y, testDataList, testLabelList)
    print('the accuracy is:', accuracy)
    print('time span:', time.time() -start)