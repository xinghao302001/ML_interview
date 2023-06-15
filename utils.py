from tqdm import tqdm
from typing import List, Union
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
def loadDataForMniST(
        dir_path:str=None
        ):
    '''
    : dir_path: the file path of MniST
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



def loadDataForText(
                    dir_path: str
                    ) -> List[str]:
    artical = []
    fr = open(dir_path, encoding="utf-8")
    for cur_line in fr.readlines():
        cur_line = cur_line.strip()
        artical.append(cur_line)
    return artical


def loadDataForIris(
                dir_path: str
                ) -> Union[List, np.ndarray]:
    
    feature_list = []  
    label_list = []  
    fr = open(dir_path)
    for line in fr.readlines():  
        cur = line.split(',')
        label = cur[-1]
        X = [float(x) for x in cur[:-1]]  #用列表来表示一条特征数据
        feature_list.append(X)
        label_list.append(label)
    FeatureArray = np.array(feature_list)  
    print('Data shape:', FeatureArray.shape)
    print('Length of labels:', len(label_list))
    return FeatureArray, label_list

def loadDataForCars(
        dir_path: str
    ) -> np.ndarray:
    df = pd.read_csv(dir_path)
    df.drop("Sports", axis=1, inplace=True)
    Xarray = np.asarray(df.values).T

    return df, Xarray




def loadDataForBBC(
        dir_path: str
    ) -> List:

    df = pd.read_csv(dir_path)
    ori_topics = df['category'].unique().tolist() ## delete duplicate
    # n = df.shape[0]

    alltexts = []
    words = []
    for text in df['text'].values:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word not in stopwords.words('english')]
        text = [word for word in text if len(word) > 3]
        alltexts.append(text)
        words.extend(set(text))
    words = list(set(words))

    return ori_topics, alltexts, words
