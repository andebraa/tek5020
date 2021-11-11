import numpy as np
import pandas as pd 

def read_data(filename):
    
    data = pd.read_csv(filename)
    print(len(data))
    print(data.head(10))
    testdata = data.iloc[lambda x: x.index % 2==0] 
    traindata = data.iloc[lambda x: x.index % 2 != 0] 
    print(len(testdata)) 
    print(testdata.head(5)) 
    print(len(traindata))
    print(traindata.head(5))

    

if __name__ == '__main__':
    read_data('ds-1.txt')
