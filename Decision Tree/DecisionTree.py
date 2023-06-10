import sys
sys.path.append(r'e:\2. work\ML_interview')

import time
import numpy as np
import sys
import os
from typing import List, Optional, Union, Dict
from utils import loadDataForMniST

class DecisionTree:
    def __init__(
                 self,
                 trainDataList: List[List[int]],
                 trainLabelList: List[int],
                 testDataList: List[List[int]],
                 testLabelList: List[int],
                 ) -> None:
        self.trainDataList = trainDataList
        self.testDataList = testDataList
        self.trainLabelList = trainLabelList
        self.testLabelList = testLabelList

    def _EmpiricalEntropy(
                          self,
                          train_labels: np.ndarray
                        ) -> float:
        '''
        compute the empiricalEntropy H_D for current labels of train dataset
        : train_labels: current label set
        : return: empiricalEntropy
        '''
        H_D = 0
        train_labelsSet = set([label for label in train_labels])
        for i in train_labelsSet:
            prob = train_labels[train_labels == i].size / train_labels.size
            H_D += -1 * prob * np.log2(prob)
        
        return H_D
    
    def _ConditinalEmpiricalEntropy(
                                    self,
                                    feature: np.ndarray,
                                    train_labels: np.ndarray
                                ) -> float:
        '''
        caculate the conditionalEmpiricalEntropy H_D_A for each feature in current train dataset
        : feature: one of the features to be caculated
        '''
        H_D_A = 0
        train_FeaturelabelSet = set([label for label in feature])
        # for each feature, caculating conditional empirical entropy
        ###　caculate H(D|A)
        for i in train_FeaturelabelSet:
            H_D_A += feature[feature == i].size / feature.size\
                        * self._EmpiricalEntropy(train_labels[feature == i])
        return H_D_A

    def _CalcBestFeature(
                        self,
                        trainDataList: List[List[int]], 
                        trainLabelList: List[int]
                        ) -> Union [float, np.ndarray] :
            trainDataArr = np.array(trainDataList)
            trainLabelArr = np.array(trainLabelList)

            featureNum = trainDataArr.shape[1]

            maxG_D_A = -1
            maxFeature = -1

            H_D = self._EmpiricalEntropy(trainLabelArr)
            for cur_feature in range(featureNum):
                trainDataArr_SplitByFeature = np.array(trainDataArr[:, cur_feature].flat)
                cur_G_D_A = H_D - self._ConditinalEmpiricalEntropy(trainDataArr_SplitByFeature, trainLabelArr)
                if cur_G_D_A > maxG_D_A:
                    maxG_D_A = cur_G_D_A
                    maxFeature = cur_feature
            return maxFeature, maxG_D_A
    
    def _getUpdateDataArr(
            self,
            trainDataArr: List[List[int]],
            trainLabelArr: List[int], 
            X: int,
            x: int,
            ) -> np.ndarray:
        '''
            update the data_feature_Set and label_set after each iteration
            : trainDataArr: dataSet required to be update
            : trainLabelArr: dataset required to be update
            : X: deleted index
            : x: when data[X] == x, the row of this sample need to be remain.
            : return: updated dataFeatureSet and labelSet
        '''
        updateDataArr = []
        updateLabelArr = []
        for i in range(len(trainDataArr)):
            if trainDataArr[i][X] == x:
                updateDataArr.append(trainDataArr[i][0:X] + trainDataArr[i][X+1:])
                updateLabelArr.append(trainLabelArr[i])

        return updateDataArr, updateLabelArr
    
    def _majorClass(
                    self,
                    labels: List[int]
                    ) -> int:
        '''
        find the most frequently label in the current dataset 

        '''
        classDict = {}
        for i in range(len(labels)):
            if labels[i] in classDict.keys():
                classDict[labels[i]] += 1
            else:
                classDict[labels[i]] = 1

        sorted_classDict = sorted(classDict.items(), key=lambda x:[1], reverse=True)

        return sorted_classDict[0][0]
    
    def _predict(
                self,
                test_data: List[List[int]],
                constructed_tree: Dict
                ):
        # for example {73: {0: {74:6}}}, if we use for to read it, its not convenient
        # dead loop, until find a class
        while True:
            (key, value), = constructed_tree.items()
            ## if current_value is dict, it means that this node is not leaf node,
            ### need to continue to tranversal
            if type(constructed_tree[key]).__name__ == "dict":
                dataVal = test_data[key]
                del test_data[key]
                constructed_tree = value[dataVal]
                if type(constructed_tree).__name__ == "int":
                    return constructed_tree
            
            else:
                return value

    def CreateTree(
                    self,
                    *dataSet:set[List],
                    epsilon: float
                   ) -> Dict:
        '''
            ID3 create tree
        : dataSet: (trainDataList, trainLabelList) 
        : return: new leaf node and its value.
        '''
        Epsilon = epsilon
        trainDataList = dataSet[0][0]
        trainLabelList = dataSet[0][1]
        print('start a node', len(trainDataList[0]), len(trainLabelList))

        classDict = {i for i in trainLabelList}
        if len(classDict) == 1:
            return trainLabelList[0]

        if len(trainDataList[0]) == 0:
            return self._majorClass(trainLabelList)
        
        maxFeature, maxInfoGain = self._CalcBestFeature(trainDataList, trainLabelList)
        if maxInfoGain < Epsilon:
            return self._majorClass(trainLabelList)

        ## Construct Trees
        treeDict = {maxFeature:{}}

        # when feature = 0, goes into "0" branch, otherwise, goes into "1" branch
        ## After splitting, return new dataArr and new labelArr
        treeDict[maxFeature][0] = self.CreateTree(self._getUpdateDataArr(trainDataList, trainLabelList, maxFeature, 0), epsilon=Epsilon)
        treeDict[maxFeature][1] = self.CreateTree(self._getUpdateDataArr(trainDataList, trainLabelList, maxFeature, 1), epsilon=Epsilon)

        return treeDict
    
    def test(
            self,
            test_data: List[List[int]],
            test_label: List[int],
            constructed_tree: Dict    
            ) -> float:
        
        correctCnt = 0
        
        for i in range(len(test_data)):
            if test_label[i] == self._predict(test_data[i], constructed_tree):
                correctCnt += 1
        
        return correctCnt / len(test_data)


if __name__ == "__main__":
    
    from utils import loadDataForMniST
    start = time.time()
    print('start read trainsSet')
    trainDataList, trainLabelList = loadDataForMniST('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadDataForMniST('e:\ML_data\mnist_test.csv')

    ### init model
    model =  DecisionTree(trainDataList, trainLabelList, testDataList, testLabelList)
    print('start to train')
    tree = model.CreateTree((trainDataList[:100], trainLabelList[:100]), epsilon=0.2)
    print('tree is:', tree)

    print('start test')
    accur = model.test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    end = time.time()
    print('time span:', end - start)
