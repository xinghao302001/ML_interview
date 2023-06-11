import sys
sys.path.append(r'e:\2. work\ML_interview')
## TODO check baum-welth 
import numpy as np
import time
from utils import loadDataForText
from typing import Optional, List

class HMMViterbi:
    def __init__(self) -> None:
        self.stateDict = {'B':0, 'I':1, 'E':2, 'S':3}

    def gettrainInitParameter(
                        self,
                        dir_path: str
                        ) -> np.ndarray:
        '''
         get statics of PI, A, B according to the article

        '''
        ## B: the begining of aphrase
        ## I: the middle word/words of a phrase
        ## E: the end of phrase
        ## O: not a phrase, just a single word

        stateDict = {'B':0, 'I':1, 'E':2, 'S':3}
        stateNum = len(stateDict)
        PI = np.zeros(stateNum)
        A = np.zeros((stateNum, stateNum))
        ## 65536 is the nuber of the observation space for Chinese word in this dataset
        B = np.zeros((stateNum, 65536))

        fr = open(dir_path, encoding="utf-8")
        ## Staticstis the appearance times of Pi, A, B
        for line in fr.readlines():
            cur_line = line.strip().split()
            wordLabels = []
            for i in range(len(cur_line)):
                if len(cur_line[i]) == 1:
                    label = "S"
                else:
                    label = "B" + "I" * (len(cur_line[i]) - 2) + "E"

                if i == 0:
                    PI[stateDict[label[0]]] += 1
                ## for each word of a phrase, generate observation state chain B
                ## and +1 for each corresponding position
                ## TODO: cur_line[i][j]: i -> the i-th phrase ; j: -> the j-th word in the i-th phrase
                for j in range(len(label)):
                    B[stateDict[label[j]]][ord(cur_line[i][j])] += 1
                
                wordLabels.extend(label)

            for i in range(1, len(wordLabels)):
                A[stateDict[wordLabels[i-1]]][stateDict[wordLabels[i]]] += 1        
            
        ## compute probability for PI, A ,B
        sumPI = np.sum(PI)
        for i in range(len(PI)):
            if PI[i] == 0:
                PI[i] = -3.14e+100
            else:
                PI[i] = np.log(PI[i] / sumPI)

        for i in range(len(A)):
            sumRow = np.sum(A[i])
            for j in range(len(A[i])):
                if A[i][j] == 0:
                    A[i][j] = -3.14e+100
                else:
                    A[i][j] = np.log(A[i][j] / sumRow)

        for i in range(len(B)):
            sumRow = np.sum(B[i])
            for j in range(len(B[i])):
                if B[i][j] == 0:
                    B[i][j] = -3.14e+100
                else:
                    B[i][j] = np.log(B[i][j] / sumRow)

        return PI, A, B
    
    def ViterbiDecoding(
                self,
                text: List[str],
                PI: np.ndarray,
                A: np.ndarray,
                B: np.ndarray
                ) -> List:
        
        decodingRes = []

        for line in text:
            delta = [[0 for i in range(len(self.stateDict))] for i in range(len(line))]
            for i in range(len(self.stateDict)):
                delta[0][i] = PI[i] + B[i][ord(line[0])]

            psi = [[0 for i in range(len(self.stateDict))] for i in range(len(line))]  

            for t in range(1, len(line)):

                for i in range(len(self.stateDict)):
                    tmpDelta = [0] * 4
                    for j in range(len(self.stateDict)):
                        tmpDelta[j] = delta[t - 1][j] + A[j][i]

                    # find the maximum δ * a
                    maxDelta = max(tmpDelta)
                    # record the state of the maximum value
                    maxDeltaIndex = tmpDelta.index(maxDelta)

                    delta[t][i] = maxDelta + B[i][ord(line[t])]
                    ## psi: used for backtrack
                    psi[t][i] = maxDeltaIndex

            sequence = []
            # obtain the index of the maximum state probability in the last state
            ## i: state i
            i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1])) 
            sequence.append(i_opt)

            for t in range(len(line)-1, 0,-1):
                i_opt = psi[t][i_opt]
                sequence.append(i_opt)
            
            sequence.reverse()

            curLine = ''
            for i in range(len(line)):
                curLine += line[i]
                if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                    curLine += '|'

            decodingRes.append(curLine)
        return decodingRes
    



if __name__ == '__main__':
    start = time.time()

    model =  HMMViterbi()
    PI, A, B = model.gettrainInitParameter('HMMTrainSet.txt')

    artical = loadDataForText('testArtical.txt')

    print('-------------------original text----------------------')
    for line in artical:
        print(line)

    partiArtical = model.ViterbiDecoding(artical, PI, A, B)

    print('-------------------after divied----------------------')
    for line in partiArtical:
        print(line)

    print('time span:', time.time() - start)
                