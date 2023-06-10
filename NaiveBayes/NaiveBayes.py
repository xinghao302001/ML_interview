import numpy as np
import time
from tqdm import tqdm
from typing import List

def loadData(
        dir_path:str=None
        ):
    '''
    : dir_path: the file path
    : return feature set and label set
    '''
    dataArr = []
    labelArr = []
    
    fr = open(dir_path)
    for line in tqdm(fr.readlines()):
        curLine = line.strip().split(',')
        # transform features of each item into binary 
        dataArr.append([int(int(num)>128) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    
    return dataArr, labelArr

def NaiveBayes(
        Py: float,
        Px_y: float,
        x) -> int:
    
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

def predict(
        Py:float, 
        Px_y: float,
        dataArr: List[List[int]],
        labelArr: List[int]
        ) -> float:
    
    correctCnt = 0
    for i in range(len(dataArr)):
        predict = NaiveBayes(Py, Px_y, dataArr[i])
        if predict == labelArr[i]:
           correctCnt += 1
    return correctCnt / len(dataArr)


def getProbability(
        dataArr: List[List[int]],
        labelArr:List[int],
        ) -> np.ndarray:
    
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

if __name__ == "__main__":
    start = time.time()
    print('start read trainsSet')
    trainDataArr, trainLabelArr = loadData('e:\ML_data\mnist_train.csv')
    print('start read testSet')
    testDataArr, testLabelArr = loadData('e:\ML_data\mnist_test.csv')
    print('start to train')
    Py, Px_y = getProbability(trainDataArr[:1000], trainLabelArr[:1000])
    print('start to test')
    accuracy = predict(Py, Px_y, testDataArr, testLabelArr)
    print('the accuracy is:', accuracy)
    print('time span:', time.time() -start)