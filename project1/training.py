import numpy as np
import pandas as pd 

def read_data(filename):
    """
    Script for reading and splitting data from datafiles.
    returns testdata and trainingdata
    """
    data = pd.read_csv(filename)
    
    #split into train and test based on odd and even index
    testdata = data.iloc[lambda x: x.index % 2==0] 
    traindata = data.iloc[lambda x: x.index % 2 != 0] 

    

if __name__ == '__main__':
    read_data('ds-1.txt')
