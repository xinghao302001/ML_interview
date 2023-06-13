import sys
sys.path.append(r'e:\2. work\ML_interview')

import time
import numpy as np
import sys
import os
from typing import List, Optional, Union, Dict
from utils import loadDataForMniST


class AdaBoost:
    def __init__(
                 self,
                 trainDataList: List[List[int]],
                 trainLabelList: List[int],
                 ) -> None:
        
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        # self.treeNum = treeNum

    def _calc_e_Gm(
                   self,
                   trainDataArr: np.ndarray,
                   trainLabelArr: np.ndarray,
                   n: int,
                   div_point: int,
                   target: int,
                   D: np.ndarray
                   ) -> Union[np.ndarray, float]:
        
        error = 0
        x = trainDataArr[:, n]
        y = trainLabelArr
        predict = []
        if target == 'LowOne':    L = 1; H = -1
        else:                   L = -1; H = 1
        
        for i in range(trainDataArr.shape[0]):
            if x[i] < div_point:
                predict.append(L)
                if y[i] != L:
                    error += D[i]
            elif x[i] >= div_point:
                predict.append(H)
                if y[i] != H:
                    error += D[i]

        return np.array(predict), error
 
    def _createSingleBoostingTree(
                                 self,
                                 trainDataArr: np.ndarray,
                                 trainLabelArr: np.ndarray,
                                 D
                                ) -> Dict:
        
        row, col = np.shape(trainDataArr)
        singleBoostTree = {}

        singleBoostTree["error"] = 1
        ### !!!!s
        for i in range(col):
            for div_point in [-0.5, 0.5 , 1.5]:
                ### !!!!!
                ## LowOne: when value is smaller than a thre. -> 1
                ## HighOne: when value is higher than a thre. -> 1
                for target in ["LowOne", "HighOne"]:
                    Gx, e = self._calc_e_Gm(trainDataArr, trainLabelArr, i, div_point, target, D)
                    if e < singleBoostTree["error"]:
                        singleBoostTree["error"] = e
                        singleBoostTree["div"] = div_point
                        singleBoostTree["rule"] = target
                        singleBoostTree["Gx"] = Gx
                        singleBoostTree["feature"] = i
                    ## TODO: ADD logic for else
        return singleBoostTree
    


    def createBoostingTree(
                           self, 
                           treeNum:int=20
                           ) -> Dict:
        
        trainDataArr = np.array(self.trainDataList)
        trainLabelArr = np.array(self.trainLabelList)
        finalpredict = [0] * len(trainLabelArr)
        row, col = np.shape(trainDataArr)

        # init D
        D = [1 / row] * row
        ### ---> # of trees = # of classifiers
        tree = []
        for i in range(treeNum):
            cur_Tree = self._createSingleBoostingTree(trainDataArr, trainLabelArr, D)
            alpha_i = 1/2 * np.log((1 - cur_Tree["error"]) / cur_Tree["error"] + 3.14e-6)
            # Gx: a vector
            ## TODO: when Gx is none?
            Gx = cur_Tree["Gx"]
            ## update D
            D = np.multiply(D, np.exp(-1 * alpha_i * np.multiply(trainLabelArr, Gx))) / sum(D)
            cur_Tree['alpha'] = alpha_i
            tree. append(cur_Tree)
            ## optional codes: used to finish in advanced.
            finalpredict += alpha_i * Gx
            error = sum([1 for i in range(len(self.trainDataList)) if np.sign(finalpredict[i]) != trainLabelArr[i]])
            finallError = error / len(self.trainDataList)
            if finallError == 0:  return tree
            print('iter:%d:%d, sigle error:%.4f, finall error:%.4f'%(i, treeNum, cur_Tree['error'], finallError ))

        return tree
    
    def _predict(self, x, div_point, target, feature):
        if target == "LowOne": 
            L = 1
            H = -1
        else:
            L = -1
            H = 1
        
        if x[feature] < div_point:
            return L
        else:
            return H
        
    def test(
             self, 
             testDataList, 
             testLabelList, 
             tree: Dict
             ):
        errorNum = 0

        for i in range(len(testDataList)):
            result = 0
            for cur_Tree in tree:
                div_point = cur_Tree["div_point"]
                target = cur_Tree["target"]
                feature = cur_Tree["feature"]
                alpha_i = cur_Tree["alpha"]
                result += alpha_i * self._predict(testDataList[i], div_point, target, feature)

            if np.sign(result) != testLabelList[i]:
                errorNum += 1
        
        return 1 - errorNum / len(testDataList)




if __name__ == "__main__":

    start = time.time()
    print('start read trainsSet')
    trainDataList, trainLabelList = loadDataForMniST('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadDataForMniST('e:\ML_data\mnist_test.csv')

    ### init model
    model = AdaBoost(trainDataList[:100], trainLabelList[:100])
    print('start to train')
    tree = model.createBoostingTree(treeNum=20)

    print('start test')
    accuracy = model.test(testDataList[:100], testLabelList[:100], tree)
    print('the accuracy is:%d' % (accuracy * 100), '%')

    end = time.time()
    print('time span:', end - start)
