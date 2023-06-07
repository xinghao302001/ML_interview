import numpy as np
import random
import math
import time
from typing import List

def CreateData(
        miu_a: float, 
        sigma_a: float, 
        miu_b: float, 
        sigma_b: float, 
        alpha_a: float, 
        alpha_b:float
        ) -> List[np.ndarray]:
    '''
        initialize two dataset that are according to two different normalization distribution
        ## TODO: add comments
    '''

    length = 1000  #data length
    data_a = np.random.normal(miu_a, sigma_a, int(length * alpha_a))
    data_b = np.random.normal(miu_b, sigma_b, int(length * alpha_b))

    ## all dataSet
    AllDataSet = []
    AllDataSet.extend(data_a)
    AllDataSet.extend(data_b)
    
    ## Shuffle AllDataset
    random.shuffle(AllDataSet)

    return AllDataSet


def CalculateGauss(
        datasetarr: np.ndarray,
        miu: float,
        sigmod: float
                   )-> np.ndarray:
        """
            Generate the distribution of the Gaussian Mixture Model 
            baased on two basic Gaussian distribution
            : datasetarr: observed data
            : miu: mean
            : sigmod: variance
        """
        result = (1 / (math.sqrt(2 * math.pi) * sigmod)) * \
             np.exp(-1 * (datasetarr - miu) * (datasetarr - miu) / (2 * sigmod**2))

        return result


def Expectation_Step(
          dataSetArr: np.ndarray, 
          alpha_a: float, 
          miu_a: float, 
          sigmod_a: float, 
          alpha_b: float, 
          miu_b: float,
          sigmod_b: float
          ) -> np.ndarray:
    """
      calculate the response weight of observation y in k-th model
      : dataSetArr: observed data
      : alpha_a: coefficients of model a
      : miu_a: mean of model a
      : sigmod_a: variance of model a
      : alpha_b: coefficients of model b
      : miu_b: mean of model b
      : sigmod_b: variance of model b
    """

    gamma_a = alpha_a * CalculateGauss(dataSetArr, miu_a, sigmod_a)
    gamma_b = alpha_b * CalculateGauss(dataSetArr, miu_b, sigmod_b)

    sumGamma = gamma_a + gamma_b
    gamma_a = gamma_a / sumGamma
    gamma_b = gamma_b / sumGamma

    return gamma_a, gamma_b


def Maximizatation_step(
        miu_a: float,
        miu_b: float,
        gamma_a: float,
        gamma_b: float,
        dataSetArr: np.ndarray
        ):
      """
       ##TODO add comments
      """
      miu_a_new = np.dot(gamma_a, dataSetArr) / np.sum(gamma_a)
      miu_b_new = np.dot(gamma_b, dataSetArr) / np.sum(gamma_b)

      sigmod_a_new = math.sqrt(np.dot(gamma_a,(dataSetArr - miu_a)**2) / np.sum(gamma_a))
      sigmod_b_new = math.sqrt(np.dot(gamma_b,(dataSetArr - miu_b)**2) / np.sum(gamma_b))

      alpha_a_new = np.sum(gamma_a) / len(gamma_a)
      alpha_b_new = np.sum(gamma_b) / len(gamma_b)

      return miu_a_new, miu_b_new, sigmod_a_new, sigmod_b_new, alpha_a_new, alpha_b_new



def EM_Train(
            dataSetList: List[np.ndarray], 
            iter: int=500
            ):
      dataSetArr = np.array(dataSetList)

      # initialization
      alpha_a, miu_a, sigmod_a = 0.5, 0, 1
      alpha_b, miu_b, sigmod_b = 0.5, 1, 1

      step  = 0
      while step < iter:
            gamma_a, gamma_b = Expectation_Step(dataSetArr, alpha_a, miu_a,sigmod_a, \
                                                alpha_b, miu_b, sigmod_b)
            miu_a_new, miu_b_new, sigmod_a_new, sigmod_b_new, alpha_a_new, alpha_b_new = Maximizatation_step(
                  miu_a, miu_b, gamma_a, gamma_b, dataSetArr
            )
            return alpha_a_new, alpha_b_new, miu_a_new, miu_b_new, sigmod_a_new, sigmod_b_new
            

if __name__ == "__main__":
    alpha_a = 0.3; miu_a = -2; sigmod_a = 0.5
    alpha_b = 0.7; miu_b = 0.5; sigmod_b = 1

    dataSetList = CreateData(miu_a, sigmod_a, miu_b, sigmod_b, alpha_a, alpha_b)
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha_a:%.1f, miu_a:%.1f, sigmoid_a:%.1f, alpha_b:%.1f, miu_b:%.1f, sigmoid_b:%.1f'%(
        alpha_a, miu_a, sigmod_a, alpha_b, miu_b, sigmod_b
    ))

    alpha_a_new, miu_a_new, sigmod_a_new, alpha_b_new, miu_b_new, sigmod_b_new = EM_Train(dataSetList)
    print('----------------------------')
    print('the Parameters predict is:')
    print('alpha_a_new:%.1f, miu_a_new:%.1f, sigmod_a_new:%.1f, alpha_b_new:%.1f, miu_b_new:%.1f, sigmod_b_new:%.1f' % (
       alpha_a_new, miu_a_new, sigmod_a_new, alpha_b_new, miu_b_new, sigmod_b_new
    ))