from tqdm import tqdm

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